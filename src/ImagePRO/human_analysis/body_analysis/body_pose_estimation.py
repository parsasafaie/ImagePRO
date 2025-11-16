from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[3]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

import cv2
import mediapipe as mp

# Constants
mp_pose = mp.solutions.pose
TOTAL_LANDMARKS = 33
DEFAULT_CONFIDENCE = 0.7
LANDMARK_RADIUS = 3
LANDMARK_COLOR = (0, 0, 255)  # Red color for landmarks

def detect_body_pose(
    image: Image,
    *,
    min_confidence: float = DEFAULT_CONFIDENCE,
    landmarks_idx: list[int] | None = None,
    pose_obj: mp.solutions.pose.Pose | None = None
) -> Result:
    """Detect body landmarks in an image using MediaPipe Pose.

    Extracts 3D body landmark coordinates and optionally draws them.
    Can focus on specific landmarks or detect all 33 points.

    Args:
        image: Input image to process.
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7
        landmarks_idx: Specific landmark indices to detect.
            If None, detects all 33 landmarks.
            Default: None
        pose_obj: Pre-initialized pose detector.
            If None, creates new instance.
            Default: None

    Returns:
        Result object with detections and visualization:
        - image: Input image with landmarks drawn
        - data: List of [idx, x, y, z] coordinates
        - meta: Operation info and parameters

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If min_confidence not in [0,1]
        TypeError: If landmarks_idx is not list[int]
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be between 0.0 and 1.0")

    if landmarks_idx is not None and (
        not isinstance(landmarks_idx, list) 
        or not all(isinstance(i, int) for i in landmarks_idx)
    ):
        raise TypeError("'landmarks_idx' must be a list of integers")

    # Initialize detector if needed
    if pose_obj is None:
        pose_obj = mp_pose.Pose(
            min_detection_confidence=min_confidence,
            static_image_mode=True
        )

    # Use all landmarks if none specified
    if landmarks_idx is None:
        landmarks_idx = list(range(TOTAL_LANDMARKS))

    # Prepare image
    h, w = image.shape[:2]
    img_copy = image._data.copy()
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    # Detect pose
    result = pose_obj.process(img_rgb)
    landmarks = []

    if result.pose_landmarks:
        # Visualize landmarks
        if len(landmarks_idx) == TOTAL_LANDMARKS:
            # Draw full skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                img_copy,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            # Draw only selected points
            for idx in landmarks_idx:
                lm = result.pose_landmarks.landmark[idx]
                x, y = int(w * lm.x), int(h * lm.y)
                cv2.circle(img_copy, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

        # Extract coordinates
        for idx in landmarks_idx:
            lm = result.pose_landmarks.landmark[idx]
            landmarks.append([idx, lm.x, lm.y, lm.z])

    return Result(
        image=img_copy,
        data=landmarks,
        meta={
            "source": image,
            "operation": "detect_body_pose",
            "min_confidence": min_confidence,
            "landmarks_idx": landmarks_idx
        }
    )


def detect_body_pose_live() -> None:
    """Start live webcam feed with real-time body pose detection.

    Opens the default camera and shows pose landmarks in real-time.
    Press ESC to exit the application.

    Raises:
        RuntimeError: If camera cannot be accessed
    """
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam")

    # Initialize pose detector in tracking mode
    pose_obj = mp_pose.Pose(
        min_detection_confidence=DEFAULT_CONFIDENCE,
        static_image_mode=False
    )

    try:
        while True:
            # Get frame
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame, skipping...")
                continue

            # Process frame
            try:
                result = detect_body_pose(
                    image=Image.from_array(frame),
                    min_confidence=DEFAULT_CONFIDENCE,
                    pose_obj=pose_obj
                )
                display_frame = result.image
            except (TypeError, ValueError):
                # Fall back to raw frame if detection fails
                display_frame = frame

            # Show result
            cv2.imshow('ImagePRO - Live Body Pose Detection', display_frame)

            # Check for ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_body_pose_live()
