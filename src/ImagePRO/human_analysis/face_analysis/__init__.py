# Face Analysis Module
# Provides facial landmark detection, pose estimation, and eye status analysis
#
# The mediapipe-based modules and the insightface-based module are imported
# guardedly so that importing the package works with only the dependencies
# a caller actually needs; unavailable modules are set to None.

try:  # requires mediapipe
    from . import eye_status_analysis
except ImportError:
    eye_status_analysis = None

try:  # requires insightface
    from . import face_comparison
except ImportError:
    face_comparison = None

try:  # requires mediapipe
    from . import face_detection
except ImportError:
    face_detection = None

try:  # requires mediapipe
    from . import face_mesh_analysis
except ImportError:
    face_mesh_analysis = None

try:  # requires mediapipe
    from . import head_pose_estimation
except ImportError:
    head_pose_estimation = None

__all__ = [
    "eye_status_analysis",
    "face_comparison",
    "face_detection",
    "face_mesh_analysis",
    "head_pose_estimation"
]
