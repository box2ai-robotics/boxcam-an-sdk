<div align="right">
  <a href="README_zh.md">中文</a> | <strong>English</strong>
</div>

# BoxCam_AN SDK

Active Near-Field Perception Camera: Active infrared + RGB night vision binocular camera

Suitable for robot wrist camera application scenarios.

## Installation

```bash
pip install .
```

## Quick Start

### Method 1: Direct Run (Display Window)

After installation, you can run directly via command line to display a window with three images:

```bash
boxcam-an
```

Or:

```bash
python -m boxcam_an.boxcam
```

This will automatically detect cameras and display a window containing three images (left to right):
1. Gray Image - RGB camera grayscale
2. RGB Image - RGB camera color
3. Near_Field - Near-field perception map (based on infrared camera)

**Controls:**
- Press 'q': Exit
- Press 'w': Increase minimum near-field perception range
- Press 's': Decrease minimum near-field perception range
- Press '+': Increase maximum near-field perception range
- Press '-': Decrease maximum near-field perception range
- Press 'f': Toggle bilateral filter
- Press 'a': Toggle adaptive normalization

### Method 2: Read Three Images Example (Display Window)

```python
from boxcam_an import BoxCam_AN
import cv2
import numpy as np

# Create and initialize cameras
boxcam_an = BoxCam_AN(auto_detect=True)
boxcam_an.initialize(rgb_width=640, rgb_height=480)

# Create display window
cv2.namedWindow('BoxCam_AN - Three Views', cv2.WINDOW_NORMAL)

try:
    while True:
        # Get RGB and infrared images
        rgb_frame, _ = boxcam_an.rgb_camera.get_latest_frame()
        ir_frame, _ = boxcam_an.infrared_camera.get_latest_frame()
        
        if rgb_frame is None or ir_frame is None:
            continue
        
        # Convert RGB to grayscale
        if len(rgb_frame.shape) == 3:
            rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        else:
            rgb_gray = rgb_frame.copy()
        
        # Convert infrared to grayscale
        if len(ir_frame.shape) == 3:
            ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        else:
            ir_gray = ir_frame.copy()
        
        # Calculate near-field perception map
        depth_map = boxcam_an.depth_estimator.estimate_depth(ir_gray)
        near_field = boxcam_an.depth_estimator.depth_to_colored(depth_map)
        
        # Convert to BGR format for display
        rgb_gray_bgr = cv2.cvtColor(rgb_gray, cv2.COLOR_GRAY2BGR) if len(rgb_gray.shape) == 2 else rgb_gray
        
        # Unify height and horizontally concatenate
        target_height = min(rgb_gray_bgr.shape[0], rgb_frame.shape[0], near_field.shape[0])
        
        def resize_keep_aspect(image, target_h):
            h, w = image.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            return cv2.resize(image, (new_w, target_h))
        
        gray_resized = resize_keep_aspect(rgb_gray_bgr, target_height)
        rgb_resized = resize_keep_aspect(rgb_frame, target_height)
        near_field_resized = resize_keep_aspect(near_field, target_height)
        
        # Horizontally concatenate three images
        combined = np.hstack([gray_resized, rgb_resized, near_field_resized])
        
        # Display image
        cv2.imshow('BoxCam_AN - Three Views', combined)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nUser interrupted")
finally:
    boxcam_an.stop()
    cv2.destroyAllWindows()
```

### Method 3: Programmatic Image Acquisition (No Display Window)

```python
from boxcam_an import BoxCam_AN
import cv2

# Create and initialize cameras
boxcam_an = BoxCam_AN(auto_detect=True)
boxcam_an.initialize(rgb_width=640, rgb_height=480)

# Get RGB and infrared images
rgb_frame, _ = boxcam_an.rgb_camera.get_latest_frame()
ir_frame, _ = boxcam_an.infrared_camera.get_latest_frame()

# Calculate near-field perception map
ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY) if len(ir_frame.shape) == 3 else ir_frame
depth_map = boxcam_an.depth_estimator.estimate_depth(ir_gray)
near_field = boxcam_an.depth_estimator.depth_to_colored(depth_map)

# Now you have three images:
# - rgb_frame: RGB color image (BGR format)
# - rgb_gray: RGB grayscale
# - near_field: Near-field perception map (BGR format)

boxcam_an.stop()
```

## Example Code

Check the `example/` directory for complete examples:
- `basic_usage.py` - Basic usage example (read three images)
- `robot_wrist_camera.py` - Robot wrist camera wrapper example

## Features

- RGB + Infrared dual camera capture
- Near-field perception based on inverse square law
- Cross-platform support (Linux/Windows/macOS)
- Automatic reconnection and error handling
- Comprehensive image validation mechanism

## Dependencies

- Python >= 3.7
- opencv-python >= 4.5.0
- numpy >= 1.19.0
