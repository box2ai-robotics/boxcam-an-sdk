#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoxCam_AN SDK Robot Wrist Camera Example
Demonstrates how to acquire and process images in robot applications
"""

import cv2
import numpy as np
from boxcam_an import BoxCam_AN


class RobotWristCamera:
    """Robot wrist camera wrapper class"""
    
    def __init__(self, rgb_camera_id=None, ir_camera_id=None):
        """
        Initialize wrist camera
        
        Args:
            rgb_camera_id: RGB camera ID (None for auto-detect)
            ir_camera_id: Infrared camera ID (None for auto-detect)
        """
        self.boxcam_an = BoxCam_AN(
            rgb_camera_id=rgb_camera_id,
            infrared_camera_id=ir_camera_id,
            auto_detect=True
        )
        self.initialized = False
    
    def start(self, rgb_width=640, rgb_height=480, ir_width=640, ir_height=480, fps=30):
        """Start camera"""
        self.initialized = self.boxcam_an.initialize(
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            ir_width=ir_width,
            ir_height=ir_height,
            fps=fps
        )
        return self.initialized
    
    def get_images(self):
        """
        Get three images
        
        Returns:
            tuple: (rgb_gray, rgb_color, near_field) or (None, None, None) if failed
                - rgb_gray: RGB grayscale (H, W)
                - rgb_color: RGB color (H, W, 3), BGR format
                - near_field: Near-field perception map (H, W, 3), BGR format
        """
        if not self.initialized:
            return None, None, None
        
        # Get RGB and IR frames
        rgb_frame, _ = self.boxcam_an.rgb_camera.get_latest_frame()
        ir_frame, _ = self.boxcam_an.infrared_camera.get_latest_frame()
        
        if rgb_frame is None or ir_frame is None:
            return None, None, None
        
        # Convert to grayscale
        if len(rgb_frame.shape) == 3:
            rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        else:
            rgb_gray = rgb_frame.copy()
        
        if len(ir_frame.shape) == 3:
            ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        else:
            ir_gray = ir_frame.copy()
        
        # Calculate near-field perception
        depth_map = self.boxcam_an.depth_estimator.estimate_depth(ir_gray)
        near_field = self.boxcam_an.depth_estimator.depth_to_colored(depth_map)
        
        return rgb_gray, rgb_frame, near_field
    
    def get_near_field_distance(self, ir_frame):
        """
        Get near-field distance map (raw distance data, unit: cm)
        
        Args:
            ir_frame: Infrared image
        
        Returns:
            numpy.ndarray: Near-field distance map (H, W), unit: cm
        """
        if len(ir_frame.shape) == 3:
            ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        else:
            ir_gray = ir_frame.copy()
        
        return self.boxcam_an.depth_estimator.estimate_depth(ir_gray)
    
    def stop(self):
        """Stop camera"""
        if self.boxcam_an:
            self.boxcam_an.stop()
        self.initialized = False


def main():
    """Example: Robot wrist camera usage"""
    camera = RobotWristCamera()
    
    # Start camera
    if not camera.start(rgb_width=640, rgb_height=480):
        print("Camera start failed")
        return
    
    print("Camera started successfully, starting to get images...")
    
    try:
        frame_count = 0
        while True:
            # Get three images
            rgb_gray, rgb_color, near_field = camera.get_images()
            
            if rgb_gray is None:
                continue
            
            frame_count += 1
            
            # Example: print information every 30 frames
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
            
            # Add your robot control logic here
            # For example:
            # - Use rgb_color for object recognition
            # - Use rgb_gray for edge detection
            # - Use near_field for obstacle avoidance
            
            # Example: detect near-field obstacles
            ir_frame, _ = camera.boxcam_an.infrared_camera.get_latest_frame()
            if ir_frame is not None:
                distance_map = camera.get_near_field_distance(ir_frame)
                min_distance = distance_map.min()
                
                if min_distance < 20:  # Obstacle within 20cm
                    print(f"Warning: Near-field obstacle detected, minimum distance: {min_distance:.1f} cm")
            
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.stop()
        print("Program ended")


if __name__ == "__main__":
    main()

