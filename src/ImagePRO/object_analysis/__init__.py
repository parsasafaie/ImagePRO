# Object Analysis Module
# Provides object detection capabilities using YOLO models

try:  # object_detection needs the optional ultralytics dependency
    from . import object_detection
except ImportError:
    object_detection = None

__all__ = ["object_detection"]
