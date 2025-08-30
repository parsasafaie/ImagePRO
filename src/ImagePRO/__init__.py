"""
ImagePRO - Professional & Modular Image Processing Library in Python

A clean, modular, and easy-to-use Python library for image processing tasks,
built with OpenCV, MediaPipe, YOLO and designed to be extensible for developers.

Whether you're working on computer vision pipelines, preprocessing images for AI models,
or simply automating batch image edits — ImagePRO gives you powerful tools with minimal effort.

Features:
- Image I/O management and batch processing
- Pre-processing: resize, crop, rotate, blur, sharpen, contrast enhancement
- Human analysis: face mesh, pose estimation, hand tracking
- Object detection with YOLO models
- Real-time processing capabilities
"""

__version__ = "1.0.0"
__author__ = "Parsa Safaie"
__email__ = "parsasafaie.2568@proton.me"

# Core modules
from . import utils
from . import pre_processing
from . import human_analysis
from . import object_analysis

__all__ = [
    "utils",
    "pre_processing", 
    "human_analysis",
    "object_analysis"
]
