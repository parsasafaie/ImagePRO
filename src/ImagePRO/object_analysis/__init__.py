# Object Analysis Module
# Provides object detection capabilities using YOLO models
#
# The ultralytics dependency is optional and imported lazily inside the
# functions that need it, so this package imports cleanly without it.

from . import object_detection

__all__ = ["object_detection"]
