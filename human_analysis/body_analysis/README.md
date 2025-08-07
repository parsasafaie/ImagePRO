# Human Body Analysis Module

This module provides tools for detecting and analyzing full-body posture and hand gestures using static images or real-time input via webcam. It leverages MediaPipe's pose and hand models for accurate 2D landmark estimation.

## Submodules

- `body_pose_estimation.py`  
  Detects 33 key body landmarks and optionally visualizes them on the image. Supports both static and real-time detection.

- `hand_tracking.py`  
  Detects 21 key hand landmarks per hand (up to 2 hands). Supports CSV export or annotated visualization. Useful for gesture recognition and interaction tasks.
