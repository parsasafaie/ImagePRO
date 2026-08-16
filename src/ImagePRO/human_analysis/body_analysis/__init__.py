# Body Analysis Module
# Provides body pose estimation and hand tracking capabilities
#
# The mediapipe dependency is optional and imported lazily inside the
# functions that need it, so both submodules are safe to import without
# any AI extras installed.

from . import body_pose_estimation
from . import hand_tracking

__all__ = ["body_pose_estimation", "hand_tracking"]
