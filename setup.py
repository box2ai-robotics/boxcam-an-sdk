#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoxCam_AN SDK Setup Script
"""

from setuptools import setup, find_packages
import os

# Read README file (if exists)
readme_file = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_file):
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements.txt
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
requirements = []
if os.path.exists(requirements_file):
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="boxcam-an",
    version="1.0.0",
    description="BoxCam_AN SDK - Active Near-Field Perception Camera",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Box2AI Team",
    author_email="",
    url="https://github.com/box2ai-robotics/boxcam-an-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video :: Capture",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "boxcam-an=boxcam_an.boxcam:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

