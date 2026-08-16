# Body Analysis Module
# Provides body pose estimation and hand tracking capabilities
#
# Both modules need the optional mediapipe dependency; they are imported
# guardedly so the package stays importable without it.

try:  # requires mediapipe
    from . import body_pose_estimation
except ImportError:
    body_pose_estimation = None

try:  # requires mediapipe
    from . import hand_tracking
except ImportError:
    hand_tracking = None

__all__ = ["body_pose_estimation", "hand_tracking"]
