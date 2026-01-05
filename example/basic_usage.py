#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoxCam_AN SDK Basic Usage Example
Suitable for robot wrist camera application scenarios

Get three images:
1. RGB grayscale
2. RGB color
3. Near-field perception map
"""

import cv2
import numpy as np
from boxcam_an import BoxCam_AN


def main():
    # Create BoxCam_AN instance (auto-detect cameras)
    boxcam_an = BoxCam_AN(auto_detect=True, camera_index=0)
    
    # Initialize cameras (can adjust resolution as needed)
    if not boxcam_an.initialize(
        rgb_width=640,
        rgb_height=480,
        ir_width=640,
        ir_height=480,
        fps=30
    ):
        print("Camera initialization failed")
        return
    
    print("Camera initialization successful, starting to get images...")
    
    try:
        # Main loop: get images and process
        for i in range(100):  # Example: get 100 frames
            # 1. Get RGB image
            rgb_frame, rgb_timestamp = boxcam_an.rgb_camera.get_latest_frame()
            if rgb_frame is None:
                continue
            
            # 2. Get infrared image
            ir_frame, ir_timestamp = boxcam_an.infrared_camera.get_latest_frame()
            if ir_frame is None:
                continue
            
            # 3. Convert RGB to grayscale
            if len(rgb_frame.shape) == 3:
                rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            else:
                rgb_gray = rgb_frame.copy()
            
            # 4. Convert infrared image to grayscale
            if len(ir_frame.shape) == 3:
                ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
            else:
                ir_gray = ir_frame.copy()
            
            # 5. Calculate near-field perception map
            depth_map = boxcam_an.depth_estimator.estimate_depth(ir_gray)
            
            # 6. Convert to colored near-field perception map
            near_field_colored = boxcam_an.depth_estimator.depth_to_colored(depth_map)
            
            # Now you have three images:
            # - rgb_gray: RGB grayscale (numpy array, shape: (H, W))
            # - rgb_frame: RGB color (numpy array, shape: (H, W, 3), BGR format)
            # - near_field_colored: Near-field perception map (numpy array, shape: (H, W, 3), BGR format)
            
            # Example: print image information
            if i % 30 == 0:  # Print every 30 frames
                print(f"Frame {i}:")
                print(f"  RGB grayscale: {rgb_gray.shape}, dtype={rgb_gray.dtype}")
                print(f"  RGB color: {rgb_frame.shape}, dtype={rgb_frame.dtype}")
                print(f"  Near-field perception map: {near_field_colored.shape}, dtype={near_field_colored.dtype}")
                print(f"  Near-field distance range: {depth_map.min():.1f} - {depth_map.max():.1f} cm")
            
            # Add your image processing logic here
            # For example: object detection, path planning, obstacle avoidance, etc.
            
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop and release resources
        boxcam_an.stop()
        print("Program ended")


if __name__ == "__main__":
    main()

