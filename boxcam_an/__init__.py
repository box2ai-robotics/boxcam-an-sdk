#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoxCam_AN SDK - Active Near-Field Perception Camera
Active infrared + RGB night vision binocular camera
Supports RGB camera, infrared camera, and near-field perception
"""

from .boxcam import BoxCam_AN, CameraCapture, DepthEstimator

__version__ = "1.0.0"
__all__ = ["BoxCam_AN", "CameraCapture", "DepthEstimator"]

