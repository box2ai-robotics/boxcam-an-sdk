#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoxCam_AN SDK - Active Near-Field Perception Camera
Active infrared + RGB night vision binocular camera
Supports RGB camera, infrared camera, and near-field perception
"""

import cv2
import time
import threading
import numpy as np
import platform
import sys
import os
import re
import glob
import logging
import subprocess
from typing import Optional, Tuple, Dict, List

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def validate_image(frame: np.ndarray, name: str = "Image") -> Tuple[bool, str]:
    """
    Validate if an image is valid
    
    Args:
        frame: Image to validate
        name: Image name (for logging)
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if frame is None:
        return False, f"{name} is None"
    
    if not isinstance(frame, np.ndarray):
        return False, f"{name} is not a numpy array"
    
    if frame.size == 0:
        return False, f"{name} size is 0"
    
    if len(frame.shape) < 2:
        return False, f"{name} has insufficient dimensions (shape={frame.shape})"
    
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return False, f"{name} has invalid size ({w}x{h})"
    
    # Check for NaN or Inf values
    if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
        return False, f"{name} contains NaN or Inf values"
    
    # Check if image is all black or all white (possibly corrupted)
    if len(frame.shape) == 2:  # Grayscale image
        if np.all(frame == 0):
            return False, f"{name} is all black (possibly corrupted)"
        if np.all(frame == 255):
            return False, f"{name} is all white (possibly corrupted)"
    elif len(frame.shape) == 3:  # Color image
        if np.all(frame == 0):
            return False, f"{name} is all black (possibly corrupted)"
        if np.all(frame == 255):
            return False, f"{name} is all white (possibly corrupted)"
    
    return True, ""


class CameraCapture:
    """Camera capture class with reconnection support and cross-platform compatibility"""
    
    def __init__(self, camera_id, name):
        self.camera_id = camera_id
        self.name = name
        self.cap = None
        self.latest_frame = None
        self.latest_timestamp = 0
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_count = 0
        self.consecutive_failures = 0
        self.last_frame_time = 0
        self.is_connected = False
        
        # Platform detection
        self.system = platform.system().lower()
        self.is_linux = self.system == 'linux'
        self.is_windows = self.system == 'windows'
        self.is_mac = self.system == 'darwin'
        
        # Select backend based on platform
        if self.is_linux:
            self.backend = cv2.CAP_V4L2
        elif self.is_windows:
            self.backend = cv2.CAP_DSHOW
        elif self.is_mac:
            self.backend = cv2.CAP_AVFOUNDATION
        else:
            self.backend = cv2.CAP_ANY
        
    def initialize(self, retry_count=3, width=640, height=480, fps=30):
        """Initialize camera with retry support"""
        for attempt in range(retry_count):
            try:
                self.cap = cv2.VideoCapture(self.camera_id, self.backend)
                
                if not self.cap.isOpened():
                    if attempt < retry_count - 1:
                        logger.warning(f"[{self.name}] Initialization failed, retrying {attempt + 1}/{retry_count}...")
                        time.sleep(0.5)
                        continue
                    logger.error(f"Error: Unable to open {self.name} camera (ID: {self.camera_id})")
                    return False
                
                # Optimize camera parameters (adjust based on platform)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Try to set MJPG format (if supported)
                if self.is_linux:
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                    except:
                        pass
                
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Set frame rate (if supported)
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, fps)
                except:
                    pass
                
                # Exposure settings (Linux specific)
                if self.is_linux:
                    try:
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
                    except:
                        pass
                
                # Warm-up: discard first few frames
                for _ in range(5):
                    self.cap.read()
                
                # Verify camera is actually working
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.cap.release()
                    if attempt < retry_count - 1:
                        logger.warning(f"[{self.name}] Unable to read test frame, retrying...")
                        continue
                    logger.error(f"[{self.name}] Unable to read test frame")
                    return False
                
                # Validate test frame
                is_valid, error_msg = validate_image(test_frame, f"{self.name} test frame")
                if not is_valid:
                    self.cap.release()
                    if attempt < retry_count - 1:
                        logger.warning(f"[{self.name}] {error_msg}, retrying...")
                        continue
                    logger.error(f"[{self.name}] {error_msg}")
                    return False
                
                self.is_connected = True
                logger.info(f"[{self.name}] Camera (ID: {self.camera_id}) initialized")
                return True
                
            except Exception as e:
                logger.error(f"[{self.name}] Initialization exception: {e}", exc_info=True)
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                if attempt < retry_count - 1:
                    time.sleep(0.5)
                    continue
                return False
        
        return False
    
    def _reconnect(self, reconnect_delay=1.0, max_retries=3):
        """Reconnect camera"""
        logger.info(f"[{self.name}] Attempting to reconnect...")
        self.is_connected = False
        
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.debug(f"[{self.name}] Error releasing camera: {e}")
            finally:
                self.cap = None
        
        time.sleep(reconnect_delay)
        
        # Try to reconnect, retry up to max_retries times
        for retry in range(max_retries):
            if self.initialize(retry_count=1):  # Each reconnection attempt only tries once, outer loop controls retries
                logger.info(f"[{self.name}] Reconnection successful")
                self.consecutive_failures = 0
                return True
            else:
                if retry < max_retries - 1:
                    logger.warning(f"[{self.name}] Reconnection failed, retrying in {reconnect_delay}s ({retry + 1}/{max_retries})...")
                    time.sleep(reconnect_delay)
        
        logger.error(f"[{self.name}] Reconnection failed (retried {max_retries} times)")
        return False
    
    def _capture_loop(self, max_consecutive_failures=50, reconnect_delay=1.0):
        """Camera capture loop (runs in separate thread)"""
        logger.info(f"[{self.name}] Capture thread started")
        
        while self.running:
            try:
                if not self.is_connected or self.cap is None:
                    time.sleep(0.1)
                    continue
                
                # Check if camera is still open
                try:
                    if not self.cap.isOpened():
                        logger.warning(f"[{self.name}] Camera unexpectedly closed")
                        self.is_connected = False
                        self._reconnect(reconnect_delay)
                        continue
                except Exception as e:
                    logger.warning(f"[{self.name}] Error checking camera status: {e}")
                    self.is_connected = False
                    self._reconnect(reconnect_delay)
                    continue
                
                ret, frame = self.cap.read()
                
                # Validate image
                if ret and frame is not None:
                    is_valid, error_msg = validate_image(frame, f"{self.name} frame")
                    if is_valid:
                        with self.frame_lock:
                            try:
                                self.latest_frame = frame.copy()
                                self.latest_timestamp = time.time()
                                self.frame_count += 1
                                self.last_frame_time = time.time()
                            except Exception as e:
                                logger.error(f"[{self.name}] Error updating frame data: {e}")
                        
                        self.consecutive_failures = 0
                        self.is_connected = True
                        
                        # Update FPS statistics
                        self.fps_counter += 1
                        elapsed = time.time() - self.fps_start_time
                        if elapsed >= 1.0:
                            self.current_fps = self.fps_counter / elapsed
                            self.fps_counter = 0
                            self.fps_start_time = time.time()
                    else:
                        # Image validation failed
                        self.consecutive_failures += 1
                        if self.consecutive_failures % 10 == 0:  # Print every 10 failures
                            logger.warning(f"[{self.name}] {error_msg} (consecutive failures: {self.consecutive_failures})")
                else:
                    # Read failed
                    self.consecutive_failures += 1
                    
                    # Check if reconnection is needed
                    if self.consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"[{self.name}] Consecutive failures: {self.consecutive_failures}, attempting reconnection...")
                        if not self._reconnect(reconnect_delay):
                            # Reconnection failed, wait and continue trying
                            time.sleep(reconnect_delay)
                    else:
                        time.sleep(0.001)
                
                # Check connection timeout (if no new frame for more than 2 seconds, consider connection lost)
                if self.is_connected and (time.time() - self.last_frame_time) > 2.0:
                    logger.warning(f"[{self.name}] Connection timeout detected, attempting reconnection...")
                    self._reconnect(reconnect_delay)
                    
            except cv2.error as e:
                logger.error(f"[{self.name}] OpenCV error: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= max_consecutive_failures:
                    self._reconnect(reconnect_delay)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"[{self.name}] Capture loop exception: {e}", exc_info=True)
                self.consecutive_failures += 1
                if self.consecutive_failures >= max_consecutive_failures:
                    self._reconnect(reconnect_delay)
                time.sleep(0.1)
    
    def start(self, max_consecutive_failures=50, reconnect_delay=1.0):
        """Start capture thread"""
        if not self.cap or not self.is_connected:
            return False
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop, 
            args=(max_consecutive_failures, reconnect_delay),
            daemon=True
        )
        self.thread.start()
        return True
    
    def stop(self):
        """Stop capture thread"""
        logger.info(f"[{self.name}] Stopping...")
        self.running = False
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(f"[{self.name}] Thread did not finish within 2 seconds")
        
        # Release camera resources
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"[{self.name}] Error releasing camera: {e}")
            finally:
                self.cap = None
        
        # Clear frame data
        with self.frame_lock:
            self.latest_frame = None
        
        self.is_connected = False
        logger.info(f"[{self.name}] Stopped")
    
    def get_latest_frame(self):
        """Get latest frame (non-blocking)"""
        try:
            with self.frame_lock:
                if self.latest_frame is not None:
                    # Validate frame again
                    is_valid, error_msg = validate_image(self.latest_frame, f"{self.name} latest frame")
                    if is_valid:
                        frame = self.latest_frame.copy()
                        return frame, self.latest_timestamp
                    else:
                        logger.debug(f"[{self.name}] {error_msg}")
                        return None, 0
            return None, 0
        except Exception as e:
            logger.error(f"[{self.name}] Error getting latest frame: {e}")
            return None, 0
    
    def get_fps(self):
        """Get current frame rate"""
        return self.current_fps
    
    def has_frame(self):
        """Check if frame is available"""
        with self.frame_lock:
            return self.latest_frame is not None and self.is_connected
    
    def get_frame_count(self):
        """Get total frame count"""
        with self.frame_lock:
            return self.frame_count
    
    def is_healthy(self):
        """Check if camera is healthy"""
        return self.is_connected and (time.time() - self.last_frame_time) < 3.0


class DepthEstimator:
    """Near-field perception estimator using inverse square law to estimate near-field distance from infrared images"""
    
    def __init__(self):
        # Inverse square law parameters (light intensity is inversely proportional to the square of distance)
        # depth = k / sqrt(intensity), where k is a calibration constant
        self.inverse_square_k = 1000.0  # Adjustable calibration constant
        
        # Near-field perception range parameters
        self.min_depth_cm = 1.0   # Minimum near-field perception distance (cm)
        self.max_depth_cm = 70.0  # Maximum near-field perception distance (cm)
        
        # Filter parameters
        self.use_bilateral_filter = True
        self.bilateral_d = 5
        self.bilateral_sigma_color = 50
        self.bilateral_sigma_space = 50
        
        # Adaptive threshold parameters
        self.adaptive_threshold = True
        self.percentile_low = 5   # Low percentile (for normalization)
        self.percentile_high = 90  # High percentile (for normalization)
        
    def intensity_to_depth_inverse_square(self, intensity):
        """
        Use inverse square law
        Assumption: light intensity is inversely proportional to the square of distance
        depth = k / sqrt(intensity)
        """
        # Avoid division by zero
        intensity = np.maximum(intensity, 1.0)
        depth = self.inverse_square_k / np.sqrt(intensity)
        return depth
    
    def radial_correction(self, gray):
        """Radial correction to compensate for lens edge brightness attenuation"""
        h, w = gray.shape
        cx, cy = w / 2, h / 2
        max_r = np.sqrt(cx**2 + cy**2)

        y, x = np.indices((h, w))
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / max_r

        # Gamma recommended between 0.6 ~ 1.0
        gamma = 0.8
        gain = 1.0 / (1.0 - gamma * r**2)

        corrected = gray.astype(np.float32) * gain
        return np.clip(corrected, 0, 255).astype(np.uint8)

    
    def estimate_depth(self, infrared_frame):
        """
        Estimate near-field distance from infrared image (using inverse square law)
        
        Args:
            infrared_frame: Infrared image (grayscale)
        
        Returns:
            depth_map: Near-field distance map (unit: cm)
        """
        # Convert to grayscale (if not already)
        if len(infrared_frame.shape) == 3:
            gray = cv2.cvtColor(infrared_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = infrared_frame.copy()
        
        # Apply bilateral filter to reduce noise (optional)
        if self.use_bilateral_filter:
            gray = cv2.bilateralFilter(gray, self.bilateral_d, 
                                      self.bilateral_sigma_color, 
                                      self.bilateral_sigma_space)
                                      
        gray = self.radial_correction(gray)
        gray = np.clip(gray, 20, 220)
        
        # Calculate near-field distance using inverse square law (needs original intensity values)
        depth_map = self.intensity_to_depth_inverse_square(gray.astype(np.float32))
        
        # Limit near-field perception range
        depth_map = np.clip(depth_map, self.min_depth_cm, self.max_depth_cm)
        
        return depth_map
    
    def depth_to_colored(self, depth_map):
        """
        Convert near-field distance map to colored visualization image
        For "Active Near-Field Perception Camera": red=near, blue=far
        
        Args:
            depth_map: Near-field distance map (unit: cm)
        
        Returns:
            depth_colored: Colored near-field perception map (BGR format)
        """
        # Adaptive normalization: use actual near-field distance distribution instead of fixed range
        # This avoids color mapping distortion when max_depth_cm is set too small
        if self.adaptive_threshold:
            # Use percentiles for adaptive normalization
            low_val = np.percentile(depth_map, self.percentile_low)
            high_val = np.percentile(depth_map, self.percentile_high)
            # Ensure high_val doesn't exceed max_depth_cm, but allow using actual distribution
            high_val = min(high_val, self.max_depth_cm)
            low_val = max(low_val, self.min_depth_cm)
            
            if high_val > low_val:
                depth_normalized = np.clip(
                    (depth_map - low_val) / (high_val - low_val) * 255, 
                    0, 255
                ).astype(np.uint8)
            else:
                # If all values are the same, use fixed range normalization
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:
            # Use fixed range normalization (original method)
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Invert depth values: make far (large values) small, near (small values) large
        # After applying JET colormap, blue=far, red=near (consistent with "Active Near-Field Perception Camera" characteristics)
        depth_normalized_inverted = 255 - depth_normalized
        
        # Apply colormap (JET: blue=far, red=near)
        depth_colored = cv2.applyColorMap(depth_normalized_inverted, cv2.COLORMAP_JET)
        
        return depth_colored


def get_device_info_linux(device_path: str) -> Dict:
    """
    Get Linux device udevadm information
    
    Args:
        device_path: Device path, e.g., /dev/video0
    
    Returns:
        dict: Dictionary containing serial, idVendor, idProduct
    """
    info = {
        'serial': None,
        'idVendor': None,
        'idProduct': None,
        'device_path': device_path
    }
    
    try:
        # Get serial
        cmd = f'udevadm info -a -n {device_path} | grep -i "ATTRS{{serial}}"'
        output = os.popen(cmd).read()
        # Extract first valid serial (usually USB device)
        matches = re.findall(r'ATTRS{serial}=="([^"]+)"', output)
        if matches:
            # Prefer serial that looks like USB device serial (exclude PCI addresses)
            for match in matches:
                if not match.startswith('0000:'):  # Exclude PCI addresses
                    info['serial'] = match
                    break
            if not info['serial'] and matches:
                info['serial'] = matches[0]
        
        # Get idVendor
        cmd = f'udevadm info -a -n {device_path} | grep -i "ATTRS{{idVendor}}"'
        output = os.popen(cmd).read()
        matches = re.findall(r'ATTRS{idVendor}=="([^"]+)"', output)
        if matches:
            # Select first non-Linux Foundation (1d6b) vendor
            for match in matches:
                if match.lower() != '1d6b':  # Linux Foundation
                    info['idVendor'] = match
                    break
            if not info['idVendor'] and matches:
                info['idVendor'] = matches[0]
        
        # Get idProduct
        cmd = f'udevadm info -a -n {device_path} | grep -i "ATTRS{{idProduct}}"'
        output = os.popen(cmd).read()
        matches = re.findall(r'ATTRS{idProduct}=="([^"]+)"', output)
        if matches:
            # Select first non-Linux Foundation product
            for match in matches:
                if match.lower() != '0002':  # Linux Foundation USB 2.0
                    info['idProduct'] = match
                    break
            if not info['idProduct'] and matches:
                info['idProduct'] = matches[0]
                
    except Exception as e:
        logger.warning(f"Error getting {device_path} information: {e}")
    
    return info


def get_device_info_windows(camera_id: int) -> Dict:
    """
    Get Windows device information (using WMI or registry)
    
    Args:
        camera_id: OpenCV camera ID
    
    Returns:
        dict: Dictionary containing serial, idVendor, idProduct
    """
    info = {
        'serial': None,
        'idVendor': None,
        'idProduct': None,
        'device_path': f'Camera {camera_id}'
    }
    
    try:
        # Method 1: Query USB device information using WMI
        try:
            import winreg
            # Try to get USB device information from registry
            # Windows registry path: HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\USB
            # This requires administrator privileges, so only basic attempt here
            pass
        except ImportError:
            pass
        
        # Method 2: Query device information using PowerShell
        try:
            ps_cmd = f'powershell -Command "Get-PnpDevice | Where-Object {{$_.InstanceId -like \'*VID_*PID_*\'}} | Select-Object InstanceId, FriendlyName"'
            output = subprocess.check_output(ps_cmd, shell=True, stderr=subprocess.DEVNULL, timeout=5).decode('utf-8', errors='ignore')
            
            # Parse output to find VID and PID
            vid_match = re.search(r'VID_([0-9A-F]{4})', output, re.IGNORECASE)
            pid_match = re.search(r'PID_([0-9A-F]{4})', output, re.IGNORECASE)
            
            if vid_match:
                info['idVendor'] = vid_match.group(1).lower()
            if pid_match:
                info['idProduct'] = pid_match.group(1).lower()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
            logger.debug(f"Windows device information query failed: {e}")
        
        # Method 3: Enumerate devices using DirectShow (if available)
        # OpenCV on Windows usually uses DirectShow backend
        # Device name can be obtained via cv2.VideoCapture.getBackendName(), but serial info is harder to get
        
    except Exception as e:
        logger.debug(f"Error getting Windows device information: {e}")
    
    return info


def get_device_info_macos(camera_id: int) -> Dict:
    """
    Get macOS device information (using system_profiler or IOKit)
    
    Args:
        camera_id: OpenCV camera ID
    
    Returns:
        dict: Dictionary containing serial, idVendor, idProduct
    """
    info = {
        'serial': None,
        'idVendor': None,
        'idProduct': None,
        'device_path': f'Camera {camera_id}'
    }
    
    try:
        # Method 1: Query USB devices using system_profiler
        try:
            cmd = 'system_profiler SPUSBDataType'
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=5).decode('utf-8', errors='ignore')
            
            # Parse output to find Vendor ID and Product ID
            vid_match = re.search(r'Vendor ID:\s*0x([0-9A-F]{4})', output, re.IGNORECASE)
            pid_match = re.search(r'Product ID:\s*0x([0-9A-F]{4})', output, re.IGNORECASE)
            serial_match = re.search(r'Serial Number:\s*([^\n]+)', output, re.IGNORECASE)
            
            if vid_match:
                info['idVendor'] = vid_match.group(1).lower()
            if pid_match:
                info['idProduct'] = pid_match.group(1).lower()
            if serial_match:
                info['serial'] = serial_match.group(1).strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
            logger.debug(f"macOS system_profiler query failed: {e}")
        
        # Method 2: Use ioreg command (more detailed information)
        try:
            cmd = 'ioreg -p IOUSB -l -w 0'
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=5).decode('utf-8', errors='ignore')
            
            # Parse ioreg output
            vid_match = re.search(r'"idVendor"\s*=\s*(\d+)', output)
            pid_match = re.search(r'"idProduct"\s*=\s*(\d+)', output)
            serial_match = re.search(r'"USB Serial Number"\s*=\s*"([^"]+)"', output)
            
            if vid_match:
                # Convert to hexadecimal
                vid_hex = format(int(vid_match.group(1)), '04x')
                info['idVendor'] = vid_hex
            if pid_match:
                pid_hex = format(int(pid_match.group(1)), '04x')
                info['idProduct'] = pid_hex
            if serial_match:
                info['serial'] = serial_match.group(1)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
            logger.debug(f"macOS ioreg query failed: {e}")
        
    except Exception as e:
        logger.debug(f"Error getting macOS device information: {e}")
    
    return info


def get_device_info(device_path: Optional[str] = None, camera_id: Optional[int] = None) -> Dict:
    """
    Get device information cross-platform
    
    Args:
        device_path: Device path (for Linux, e.g., /dev/video0)
        camera_id: Camera ID (for Windows/macOS)
    
    Returns:
        dict: Dictionary containing serial, idVendor, idProduct
    """
    system = platform.system().lower()
    
    if system == 'linux':
        if device_path:
            return get_device_info_linux(device_path)
        else:
            logger.warning("Linux platform requires device_path parameter")
            return {
                'serial': None,
                'idVendor': None,
                'idProduct': None,
                'device_path': device_path or 'unknown'
            }
    elif system == 'windows':
        if camera_id is not None:
            return get_device_info_windows(camera_id)
        else:
            logger.warning("Windows platform requires camera_id parameter")
            return {
                'serial': None,
                'idVendor': None,
                'idProduct': None,
                'device_path': f'Camera {camera_id or "unknown"}'
            }
    elif system == 'darwin':  # macOS
        if camera_id is not None:
            return get_device_info_macos(camera_id)
        else:
            logger.warning("macOS platform requires camera_id parameter")
            return {
                'serial': None,
                'idVendor': None,
                'idProduct': None,
                'device_path': f'Camera {camera_id or "unknown"}'
            }
    else:
        logger.warning(f"Unknown platform: {system}")
        return {
            'serial': None,
            'idVendor': None,
            'idProduct': None,
            'device_path': device_path or f'Camera {camera_id or "unknown"}'
        }


def test_opencv_camera(camera_id):
    """
    Cross-platform test if OpenCV can open camera with specified ID
    
    Args:
        camera_id: OpenCV camera ID
    
    Returns:
        tuple: (success, device_path) or (False, None)
    """
    try:
        system = platform.system().lower()
        
        # Select backend based on platform
        if system == 'linux':
            backend = cv2.CAP_V4L2
        elif system == 'windows':
            backend = cv2.CAP_DSHOW
        elif system == 'darwin':  # macOS
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_ANY
        
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            # Try to read a frame to confirm camera is actually available
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                # Validate frame
                is_valid, _ = validate_image(frame, f"test frame {camera_id}")
                if is_valid:
                    # Return device path based on platform
                    if system == 'linux':
                        device_path = f'/dev/video{camera_id}'
                    elif system == 'windows':
                        device_path = f'Camera {camera_id}'
                    elif system == 'darwin':
                        device_path = f'Camera {camera_id}'
                    else:
                        device_path = f'Camera {camera_id}'
                    return True, device_path
        return False, None
    except Exception as e:
        logger.debug(f"Error testing camera {camera_id}: {e}")
        return False, None


def scan_video_devices(max_camera_id=20):
    """
    Cross-platform scan all video devices and match OpenCV IDs
    
    Args:
        max_camera_id: Maximum number of camera IDs to test
    
    Returns:
        list: List of device information
    """
    system = platform.system().lower()
    is_linux = system == 'linux'
    is_windows = system == 'windows'
    is_mac = system == 'darwin'
    
    devices_info = []
    
    # Linux: Find all /dev/video* devices
    if is_linux:
        try:
            video_devices = sorted(glob.glob('/dev/video*'))
            
            # Get information for each device file
            for device_path in video_devices:
                info = get_device_info(device_path=device_path)
                devices_info.append(info)
        except Exception as e:
            logger.warning(f"Error scanning device files: {e}")
    
    # Windows/macOS: Test camera IDs directly via OpenCV
    # Test OpenCV camera IDs (usually starting from 0)
    results = []
    
    for camera_id in range(max_camera_id):
        success, device_path = test_opencv_camera(camera_id)
        if success:
            # Match device information
            matched_device = None
            
            if is_linux:
                # Linux: Match from already scanned device list
                for dev_info in devices_info:
                    if dev_info['device_path'] == device_path:
                        matched_device = dev_info
                        break
                
                # If not matched, try to get device information directly
                if not matched_device and device_path:
                    matched_device = get_device_info(device_path=device_path)
            elif is_windows or is_mac:
                # Windows/macOS: Get device information directly via camera ID
                matched_device = get_device_info(camera_id=camera_id)
            
            # If still not matched, create a basic device information
            if not matched_device:
                if is_linux:
                    device_path_str = device_path if device_path else f'/dev/video{camera_id}'
                else:
                    device_path_str = f'Camera {camera_id}'
                
                matched_device = {
                    'device_path': device_path_str,
                    'serial': None,
                    'idVendor': None,
                    'idProduct': None
                }
            
            result = {
                'opencv_id': camera_id,
                'device_info': matched_device
            }
            results.append(result)
    
    return results


def detect_camera_pairs(results, target_vendor='2c7f'):
    """
    Detect camera pairs (two lenses of the same camera)
    One camera has two lenses: USB2.0_CAM1 and USB2.0_CAM2
    
    Camera ID mapping rules:
    - USB2.0_CAM1 (RGB camera) -> OpenCV ID 4
    - USB2.0_CAM2 (IR camera) -> OpenCV ID 2
    
    Args:
        results: Scan result list
        target_vendor: Target Vendor ID (default 2c7f)
    
    Returns:
        list: Camera pair list, each element contains {'rgb_id': int, 'ir_id': int, 'pair_index': int}
    """
    camera_pairs = []
    
    # Find all matching cameras, sorted by ID
    rgb_cameras = []  # USB2.0_CAM1 -> usually OpenCV ID 4
    ir_cameras = []   # USB2.0_CAM2 -> usually OpenCV ID 2
    
    for result in results:
        opencv_id = result['opencv_id']
        dev_info = result['device_info']
        
        vendor = str(dev_info.get('idVendor', '')).lower().strip()
        serial = str(dev_info.get('serial', '')).strip()
        
        # Check if it's the target Vendor
        if vendor == target_vendor.lower():
            if serial == 'USB2.0_CAM1':
                rgb_cameras.append(opencv_id)
            elif serial == 'USB2.0_CAM2':
                ir_cameras.append(opencv_id)
    
    # Sort for better matching
    rgb_cameras.sort()
    ir_cameras.sort()
    
    # Match camera pairs: find RGB and IR cameras with connected IDs
    # Usually the two lenses of the same camera have connected IDs (e.g., 2 and 4, or 4 and 6)
    # Matching strategy: for each RGB camera, find the nearest IR camera (ID difference of 2)
    used_ir_ids = set()
    
    for rgb_id in rgb_cameras:
        best_match = None
        min_diff = float('inf')
        
        for ir_id in ir_cameras:
            if ir_id in used_ir_ids:
                continue
            
            diff = abs(rgb_id - ir_id)
            # Prefer matching IDs with difference of 2 (two lenses of same camera)
            if diff == 2:
                best_match = ir_id
                min_diff = diff
                break
            elif diff < min_diff:
                best_match = ir_id
                min_diff = diff
        
        if best_match is not None:
            used_ir_ids.add(best_match)
            # Determine RGB and IR based on serial: USB2.0_CAM1 is RGB, USB2.0_CAM2 is IR
            # Note: IR ID (2) may be smaller than RGB ID (4)
            camera_pairs.append({
                'rgb_id': rgb_id,  # USB2.0_CAM1
                'ir_id': best_match,  # USB2.0_CAM2
                'pair_index': len(camera_pairs)
            })
    
    return camera_pairs


def auto_detect_cameras(interactive=True, camera_index=0):
    """
    Automatically detect and select cameras
    
    Args:
        interactive: Whether to interactively select (if multiple cameras)
        camera_index: If non-interactive, select which camera pair (starting from 0)
    
    Returns:
        tuple: (rgb_camera_id, infrared_camera_id) or (None, None) if not found
    """
    logger.info("Scanning camera devices...")
    results = scan_video_devices(max_camera_id=20)
    
    if not results:
        logger.error("No available cameras found")
        return None, None
    
    # Detect camera pairs
    camera_pairs = detect_camera_pairs(results)
    
    if not camera_pairs:
        logger.warning("No matching camera pairs found (Vendor=2c7f, Serial=USB2.0_CAM1/CAM2)")
        logger.info("Tip: If your camera uses different Vendor ID or Serial, please manually specify camera IDs")
        return None, None
    
    logger.info(f"\nFound {len(camera_pairs)} camera pair(s):")
    print("=" * 60)
    for i, pair in enumerate(camera_pairs):
        print(f"Camera pair {i+1}: RGB ID={pair['rgb_id']}, IR ID={pair['ir_id']}")
    print("=" * 60)
    
    # If multiple camera pairs, let user choose
    if len(camera_pairs) > 1:
        if interactive:
            while True:
                try:
                    choice = input(f"\nPlease select camera pair to use (1-{len(camera_pairs)}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(camera_pairs):
                        selected_pair = camera_pairs[choice_idx]
                        print(f"\nSelected camera pair {choice}: RGB ID={selected_pair['rgb_id']}, IR ID={selected_pair['ir_id']}")
                        return selected_pair['rgb_id'], selected_pair['ir_id']
                    else:
                        print(f"Invalid selection, please enter a number between 1-{len(camera_pairs)}")
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nUser cancelled")
                    return None, None
        else:
            # Non-interactive, use specified index
            if 0 <= camera_index < len(camera_pairs):
                selected_pair = camera_pairs[camera_index]
                print(f"\nUsing camera pair {camera_index+1}: RGB ID={selected_pair['rgb_id']}, IR ID={selected_pair['ir_id']}")
                return selected_pair['rgb_id'], selected_pair['ir_id']
            else:
                print(f"Error: Camera index {camera_index} out of range (0-{len(camera_pairs)-1})")
                return None, None
    else:
        # Only one camera pair, use directly
        selected_pair = camera_pairs[0]
        print(f"\nUsing camera pair: RGB ID={selected_pair['rgb_id']}, IR ID={selected_pair['ir_id']}")
        return selected_pair['rgb_id'], selected_pair['ir_id']


class BoxCam_AN:
    """BoxCam_AN main class integrating camera capture and near-field perception functionality"""
    
    def __init__(self, rgb_camera_id=None, infrared_camera_id=None, auto_detect=True, camera_index=0):
        """
        Initialize BoxCam_AN system
        
        Camera ID mapping rules (default values):
        - rgb_camera_id: 4 (corresponds to USB2.0_CAM1)
        - infrared_camera_id: 2 (corresponds to USB2.0_CAM2)
        
        Args:
            rgb_camera_id: RGB camera ID (if None and auto_detect=True, auto-detect)
            infrared_camera_id: Infrared camera ID (if None and auto_detect=True, auto-detect)
            auto_detect: Whether to auto-detect cameras (default True)
            camera_index: If multiple camera pairs, select which one (starting from 0, default 0)
        """
        # If auto-detect is enabled and camera IDs are not specified, auto-detect
        if auto_detect and (rgb_camera_id is None or infrared_camera_id is None):
            detected_rgb_id, detected_ir_id = auto_detect_cameras(interactive=True, camera_index=camera_index)
            if detected_rgb_id is None or detected_ir_id is None:
                raise RuntimeError("Auto-detection failed, please manually specify camera IDs")
            self.rgb_camera_id = detected_rgb_id
            self.infrared_camera_id = detected_ir_id
        else:
            # Use specified camera IDs, or default values if None
            # Default mapping: RGB=4 (USB2.0_CAM1), IR=2 (USB2.0_CAM2)
            self.rgb_camera_id = rgb_camera_id if rgb_camera_id is not None else 4
            self.infrared_camera_id = infrared_camera_id if infrared_camera_id is not None else 2
        
        logger.info(f"[BoxCam_AN] Camera ID mapping: RGB={self.rgb_camera_id} (USB2.0_CAM1), IR={self.infrared_camera_id} (USB2.0_CAM2)")
        
        self.rgb_camera = None
        self.infrared_camera = None
        self.depth_estimator = DepthEstimator()
        
        self.running = False
        self.target_fps = 30.0
        self.frame_time = 1.0 / self.target_fps
        
    def initialize(self, rgb_width=1920, rgb_height=1080, ir_width=1920, ir_height=1080, fps=30):
        """
        Initialize cameras
        
        Args:
            rgb_width: RGB camera width (supported: 640, 800, 1280, 1920, default 1920)
            rgb_height: RGB camera height (supported: 480, 600, 720, 1080, default 1080)
            ir_width: Infrared camera width (supported: 640, 800, 1280, 1920, default 1920)
            ir_height: Infrared camera height (supported: 480, 600, 720, 1080, default 1080)
            fps: Target frame rate (default 30)
        
        Supported size combinations:
            - 640x480
            - 800x600
            - 1280x720
            - 1920x1080
        """
        # Validate if sizes are supported
        supported_sizes = [
            (640, 480), (800, 600), (1280, 720), (1920, 1080)
        ]
        
        rgb_size = (rgb_width, rgb_height)
        ir_size = (ir_width, ir_height)
        
        if rgb_size not in supported_sizes:
            logger.warning(f"RGB camera size {rgb_size} not in supported list, will try to use")
        if ir_size not in supported_sizes:
            logger.warning(f"Infrared camera size {ir_size} not in supported list, will try to use")
        
        self.target_fps = fps
        self.frame_time = 1.0 / fps
        
        # Initialize RGB camera (USB2.0_CAM1, default ID=4)
        try:
            self.rgb_camera = CameraCapture(self.rgb_camera_id, "RGB")
            logger.info(f"[BoxCam_AN] Initializing RGB camera (ID={self.rgb_camera_id}, USB2.0_CAM1)...")
            if not self.rgb_camera.initialize(retry_count=3, width=rgb_width, height=rgb_height, fps=fps):
                logger.error(f"RGB camera (ID={self.rgb_camera_id}) initialization failed")
                return False
        except Exception as e:
            logger.error(f"Error creating RGB camera object: {e}", exc_info=True)
            return False
        
        # Initialize infrared camera (USB2.0_CAM2, default ID=2)
        try:
            self.infrared_camera = CameraCapture(self.infrared_camera_id, "IR")
            logger.info(f"[BoxCam_AN] Initializing IR camera (ID={self.infrared_camera_id}, USB2.0_CAM2)...")
            if not self.infrared_camera.initialize(retry_count=3, width=ir_width, height=ir_height, fps=fps):
                logger.error(f"Infrared camera (ID={self.infrared_camera_id}) initialization failed")
                if self.rgb_camera:
                    self.rgb_camera.stop()
                return False
        except Exception as e:
            logger.error(f"Error creating IR camera object: {e}", exc_info=True)
            if self.rgb_camera:
                self.rgb_camera.stop()
            return False
        
        # Start capture threads
        try:
            if not self.rgb_camera.start():
                logger.error("RGB camera thread start failed")
                self.rgb_camera.stop()
                self.infrared_camera.stop()
                return False
        except Exception as e:
            logger.error(f"Error starting RGB camera thread: {e}", exc_info=True)
            self.rgb_camera.stop()
            self.infrared_camera.stop()
            return False
        
        try:
            if not self.infrared_camera.start():
                logger.error("Infrared camera thread start failed")
                self.rgb_camera.stop()
                self.infrared_camera.stop()
                return False
        except Exception as e:
            logger.error(f"Error starting IR camera thread: {e}", exc_info=True)
            self.rgb_camera.stop()
            self.infrared_camera.stop()
            return False
        
        # Wait for cameras to stabilize and verify data output
        logger.info("\nWaiting for cameras to capture frames...")
        max_wait_time = 5.0
        wait_start = time.time()
        frames_ready = False
        while True:
            if self.infrared_camera.has_frame() and self.rgb_camera.has_frame():
                frames_ready = True
                logger.info("✓ Both cameras are ready")
                break
            if time.time() - wait_start > max_wait_time:
                logger.warning("\nWait timeout, continuing...")
                break
            time.sleep(0.1)
            print(".", end='', flush=True)
        
        # Verify camera output data format
        if frames_ready:
            try:
                ir_frame, _ = self.infrared_camera.get_latest_frame()
                rgb_frame, _ = self.rgb_camera.get_latest_frame()
                
                if rgb_frame is not None:
                    rgb_shape = rgb_frame.shape
                    if len(rgb_shape) == 3 and rgb_shape[2] == 3:
                        logger.info(f"✓ RGB camera (ID={self.rgb_camera_id}) outputs color image: {rgb_shape}")
                    else:
                        logger.warning(f"RGB camera (ID={self.rgb_camera_id}) outputs non-color image: {rgb_shape}")
                        logger.info(f"  Expected: 3-channel color image (H, W, 3)")
                
                if ir_frame is not None:
                    ir_shape = ir_frame.shape
                    logger.info(f"✓ IR camera (ID={self.infrared_camera_id}) outputs image: {ir_shape}")
            except Exception as e:
                logger.warning(f"Error verifying camera output format: {e}")
        
        logger.info("\nSystem ready")
        return True
    
    def _resize_keep_aspect_ratio(self, image, target_height):
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            target_height: Target height
        
        Returns:
            Resized image
        """
        if image is None or image.size == 0:
            return None
        
        h, w = image.shape[:2]
        if h == 0:
            return None
        
        # Calculate scale factor
        scale = target_height / h
        new_width = int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def run(self):
        """Run main loop, display three images in a single window: grayscale, RGB, near-field perception"""
        if not self.rgb_camera or not self.infrared_camera:
            print("Error: Cameras not initialized")
            return
        
        self.running = True
        
        print("=" * 60)
        print("BoxCam_AN SDK - Active Near-Field Perception Camera")
        print("=" * 60)
        print("Display Window:")
        print("  Unified window displaying three images (left to right):")
        print("  1. Gray Image - RGB camera grayscale")
        print("  2. RGB Image - RGB camera color")
        print("  3. Near_Field - Near-field perception map (based on infrared camera)")
        print("=" * 60)
        print("Controls:")
        print("  Press 'q': Exit")
        print("  Press 'w': Increase minimum near-field perception range")
        print("  Press 's': Decrease minimum near-field perception range")
        print("  Press '+': Increase maximum near-field perception range")
        print("  Press '-': Decrease maximum near-field perception range")
        print("  Press 'f': Toggle bilateral filter")
        print("  Press 'a': Toggle adaptive normalization")
        print("  Tip: Window supports mouse drag to resize, three images scale uniformly")
        print("=" * 60)
        
        # Create unified window and set default size
        cv2.namedWindow('BoxCam_AN - Three Views', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('BoxCam_AN - Three Views', 1920, 420)
        
        try:
            while self.running:
                frame_start_time = time.time()
                
                # Read frames
                infrared_frame, infrared_timestamp = self.infrared_camera.get_latest_frame()
                rgb_frame, rgb_timestamp = self.rgb_camera.get_latest_frame()
                
                # Check frame availability
                if infrared_frame is None:
                    time.sleep(0.01)
                    continue
                
                if rgb_frame is None:
                    time.sleep(0.01)
                    continue
                
                # Data validation: use validate_image function for complete validation
                rgb_valid, rgb_error = validate_image(rgb_frame, "RGB frame")
                if not rgb_valid:
                    logger.warning(f"RGB frame validation failed: {rgb_error}")
                    time.sleep(0.01)
                    continue
                
                ir_valid, ir_error = validate_image(infrared_frame, "IR frame")
                if not ir_valid:
                    logger.warning(f"IR frame validation failed: {ir_error}")
                    time.sleep(0.01)
                    continue
                
                # Convert RGB to grayscale (for display)
                # RGB image should be color BGR format (3 channels), if not it might be IR image
                try:
                    if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
                        rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        # If RGB frame is not 3-channel, might be IR image, issue warning
                        logger.warning(f"RGB camera (ID={self.rgb_camera_id}) output is not color image (shape={rgb_frame.shape})")
                        rgb_gray = rgb_frame.copy() if len(rgb_frame.shape) == 2 else cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                except Exception as e:
                    logger.error(f"Error converting RGB to grayscale: {e}")
                    time.sleep(0.01)
                    continue
                
                # Convert infrared image to grayscale (for near-field perception)
                # IR image is usually single-channel grayscale, but might be BGR format
                try:
                    if len(infrared_frame.shape) == 3:
                        infrared_gray = cv2.cvtColor(infrared_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        infrared_gray = infrared_frame.copy()
                except Exception as e:
                    logger.error(f"Error converting IR to grayscale: {e}")
                    time.sleep(0.01)
                    continue
                
                # Estimate depth (using infrared image)
                try:
                    depth_map = self.depth_estimator.estimate_depth(infrared_gray)
                    if depth_map is None or depth_map.size == 0:
                        logger.warning("Near-field perception returned empty result")
                        time.sleep(0.01)
                        continue
                except Exception as e:
                    logger.error(f"Error in near-field perception: {e}")
                    time.sleep(0.01)
                    continue
                
                # Convert to colored near-field perception map (Near_Field perception map)
                try:
                    near_field_colored = self.depth_estimator.depth_to_colored(depth_map)
                    if near_field_colored is None or near_field_colored.size == 0:
                        logger.warning("Near-field perception map coloring returned empty result")
                        time.sleep(0.01)
                        continue
                except Exception as e:
                    logger.error(f"Error coloring near-field perception map: {e}")
                    time.sleep(0.01)
                    continue
                
                # Get frame rates
                infrared_fps = self.infrared_camera.get_fps()
                rgb_fps = self.rgb_camera.get_fps()
                
                # Prepare display frames (ensure BGR format)
                try:
                    # Gray Image displays RGB camera grayscale
                    gray_display = cv2.cvtColor(rgb_gray, cv2.COLOR_GRAY2BGR) if len(rgb_gray.shape) == 2 else rgb_gray.copy()
                    rgb_display = rgb_frame.copy()
                    near_field_display = near_field_colored.copy()
                    
                    # Add information to images (including camera ID info for debugging)
                    rgb_info = f"RGB Gray (ID={self.rgb_camera_id}) | FPS: {rgb_fps:.1f}Hz"
                    cv2.putText(gray_display, rgb_info, 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    rgb_color_info = f"RGB Image (ID={self.rgb_camera_id}) | FPS: {rgb_fps:.1f}Hz"
                    cv2.putText(rgb_display, rgb_color_info, 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    near_field_info = f"Near Field: {self.depth_estimator.min_depth_cm:.0f}-{self.depth_estimator.max_depth_cm:.0f}cm"
                    ir_info = f"Near_Field (IR ID={self.infrared_camera_id}) | FPS: {infrared_fps:.1f}Hz"
                    cv2.putText(near_field_display, ir_info, 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(near_field_display, near_field_info, 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Unify height of three images (use minimum height, maintain aspect ratio)
                    heights = [gray_display.shape[0], rgb_display.shape[0], near_field_display.shape[0]]
                    target_height = min(heights)
                    
                    # Resize each image, maintain aspect ratio
                    gray_resized = self._resize_keep_aspect_ratio(gray_display, target_height)
                    rgb_resized = self._resize_keep_aspect_ratio(rgb_display, target_height)
                    near_field_resized = self._resize_keep_aspect_ratio(near_field_display, target_height)
                    
                    # Check if resized images are valid
                    if gray_resized is None or rgb_resized is None or near_field_resized is None:
                        logger.warning("Image resize failed")
                        time.sleep(0.01)
                        continue
                    
                    # Validate resized images
                    gray_valid, _ = validate_image(gray_resized, "resized grayscale")
                    rgb_valid, _ = validate_image(rgb_resized, "resized RGB")
                    near_field_valid, _ = validate_image(near_field_resized, "resized near-field")
                    
                    if not (gray_valid and rgb_valid and near_field_valid):
                        logger.warning("Resized image validation failed")
                        time.sleep(0.01)
                        continue
                    
                    # Horizontally concatenate three images
                    try:
                        combined_image = np.hstack([gray_resized, rgb_resized, near_field_resized])
                        
                        # Validate concatenated image
                        combined_valid, combined_error = validate_image(combined_image, "concatenated image")
                        if not combined_valid:
                            logger.warning(f"Concatenated image validation failed: {combined_error}")
                            time.sleep(0.01)
                            continue
                        
                        # Display in unified window
                        cv2.imshow('BoxCam_AN - Three Views', combined_image)
                    except Exception as e:
                        logger.error(f"Error concatenating or displaying image: {e}")
                        time.sleep(0.01)
                        continue
                except Exception as e:
                    logger.error(f"Error displaying image: {e}", exc_info=True)
                    time.sleep(0.01)
                    continue
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('w'):
                    self.depth_estimator.min_depth_cm = min(
                        self.depth_estimator.max_depth_cm - 1.0, 
                        self.depth_estimator.min_depth_cm + 10.0
                    )
                    print(f"\nMinimum near-field perception range increased to: {self.depth_estimator.min_depth_cm:.0f} cm")
                elif key == ord('s'):
                    self.depth_estimator.min_depth_cm = max(
                        1.0, 
                        self.depth_estimator.min_depth_cm - 10.0
                    )
                    print(f"\nMinimum near-field perception range decreased to: {self.depth_estimator.min_depth_cm:.0f} cm")
                elif key == ord('+') or key == ord('='):
                    self.depth_estimator.max_depth_cm += 10.0
                    print(f"\nMaximum near-field perception range increased to: {self.depth_estimator.max_depth_cm:.0f} cm")
                elif key == ord('-') or key == ord('_'):
                    self.depth_estimator.max_depth_cm = max(
                        self.depth_estimator.min_depth_cm + 1.0, 
                        self.depth_estimator.max_depth_cm - 10.0
                    )
                    print(f"\nMaximum near-field perception range decreased to: {self.depth_estimator.max_depth_cm:.0f} cm")
                elif key == ord('f'):
                    self.depth_estimator.use_bilateral_filter = not self.depth_estimator.use_bilateral_filter
                    print(f"\nBilateral filter: {'enabled' if self.depth_estimator.use_bilateral_filter else 'disabled'}")
                elif key == ord('a'):
                    self.depth_estimator.adaptive_threshold = not self.depth_estimator.adaptive_threshold
                    print(f"\nAdaptive threshold: {'enabled' if self.depth_estimator.adaptive_threshold else 'disabled'}")
                
                # Frame rate locking
                frame_end_time = time.time()
                frame_elapsed = frame_end_time - frame_start_time
                sleep_time = self.frame_time - frame_elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("\n\nInterrupt signal received...")
        except Exception as e:
            logger.error(f"\n\nMain loop exception: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop and release resources"""
        self.running = False
        logger.info("\nClosing cameras...")
        
        try:
            if self.rgb_camera:
                self.rgb_camera.stop()
        except Exception as e:
            logger.error(f"Error stopping RGB camera: {e}")
        
        try:
            if self.infrared_camera:
                self.infrared_camera.stop()
        except Exception as e:
            logger.error(f"Error stopping IR camera: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"Error closing windows: {e}")
        
        logger.info("Program exited")
    
    def get_depth_estimator(self):
        """Get depth estimator instance (for adjusting parameters)"""
        return self.depth_estimator
    
    def get_rgb_camera(self):
        """Get RGB camera instance"""
        return self.rgb_camera
    
    def get_infrared_camera(self):
        """Get infrared camera instance"""
        return self.infrared_camera


def main():
    """Main function entry point"""
    import sys
    
    print("=" * 60)
    print("BoxCam_AN SDK - Active Near-Field Perception Camera")
    print("=" * 60)
    
    # Create BoxCam_AN instance (auto-detect cameras)
    try:
        boxcam_an = BoxCam_AN(auto_detect=True, camera_index=0)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTip: If auto-detection fails, you can manually specify camera IDs:")
        print("  boxcam_an = BoxCam_AN(rgb_camera_id=4, infrared_camera_id=2, auto_detect=False)")
        return 1
    
    # Initialize cameras
    if not boxcam_an.initialize():
        print("Initialization failed, exiting program")
        return 1
    
    # Run main loop
    try:
        boxcam_an.run()
    except Exception as e:
        print(f"\nRuntime exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
