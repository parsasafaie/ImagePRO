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
mp_hands = mp.solutions.hands
DEFAULT_MAX_HANDS = 2
DEFAULT_MIN_CONFIDENCE = 0.7
TOTAL_HAND_LANDMARKS = 21
LANDMARK_RADIUS = 3
LANDMARK_COLOR = (0, 0, 255)  # Red color for landmarks


def detect_hands(
    image: Image,
    *,
    max_hands: int = DEFAULT_MAX_HANDS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    landmarks_idx: list[int] | None = None,
    hands_obj: mp.solutions.hands.Hands | None = None
) -> Result:
    """Detect hand landmarks in an image using MediaPipe Hands.

    Identifies hand positions and extracts 3D coordinates of
    key points (joints, fingertips, etc.).

    Args:
        image: Input image to process.
        max_hands: Maximum number of hands to detect.
            Must be positive.
            Default: 2
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7
        landmarks_idx: Specific landmark indices to detect.
            If None, detects all 21 points per hand.
            Default: None
        hands_obj: Pre-initialized hand detector.
            If None, creates new instance.
            Default: None

    Returns:
        Result object with detections and visualization:
        - image: Input image with landmarks drawn
        - data: List of [hand_id, point_id, x, y, z]
        - meta: Operation info and parameters

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If max_hands is not positive
        ValueError: If min_confidence not in [0,1]
        TypeError: If landmarks_idx is not list[int]
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(max_hands, int) or max_hands <= 0:
        raise ValueError("'max_hands' must be positive")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be between 0.0 and 1.0")

    if landmarks_idx is not None and (
        not isinstance(landmarks_idx, list) 
        or not all(isinstance(i, int) for i in landmarks_idx)
    ):
        raise TypeError("'landmarks_idx' must be a list of integers")

    # Initialize detector if needed
    if hands_obj is None:
        hands_obj = mp_hands.Hands(
            min_detection_confidence=min_confidence,
            max_num_hands=max_hands,
            static_image_mode=True
        )

    # Use all landmarks if none specified
    if landmarks_idx is None:
        landmarks_idx = list(range(TOTAL_HAND_LANDMARKS))

    # Prepare image
    img_copy = image._data.copy()
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands_obj.process(img_rgb)
    landmarks = []

    # Process detections
    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Visualize landmarks
            if len(landmarks_idx) == TOTAL_HAND_LANDMARKS:
                # Draw full hand skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img_copy,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
            else:
                # Draw only selected points
                h, w, _ = img_copy.shape
                for idx in landmarks_idx:
                    lm = hand_landmarks.landmark[idx]
                    x, y = int(w * lm.x), int(h * lm.y)
                    cv2.circle(img_copy, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

            # Extract coordinates
            for idx in landmarks_idx:
                lm = hand_landmarks.landmark[idx]
                landmarks.append([hand_id, idx, lm.x, lm.y, lm.z])

    return Result(
        image=img_copy,
        data=landmarks,
        meta={
            "source": image,
            "operation": "detect_hands",
            "max_hands": max_hands,
            "min_confidence": min_confidence,
            "landmarks_idx": landmarks_idx
        }
    )


def detect_hands_live(
    max_hands: int = DEFAULT_MAX_HANDS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> None:
    """Start live webcam feed with real-time hand detection.

    Opens the default camera and shows hand landmarks in real-time.
    Press ESC to exit.

    Args:
        max_hands: Maximum number of hands to track.
            Must be positive.
            Default: 2
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If camera cannot be accessed
    """
    # Validate parameters
    if not isinstance(max_hands, int) or max_hands <= 0:
        raise ValueError("'max_hands' must be positive")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be between 0.0 and 1.0")

    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam")

    # Initialize hand detector in tracking mode
    hands_obj = mp_hands.Hands(
        min_detection_confidence=min_confidence,
        max_num_hands=max_hands,
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
            result = detect_hands(
                image=Image.from_array(frame),
                max_hands=max_hands,
                min_confidence=min_confidence,
                hands_obj=hands_obj
            )
            display_frame = result.image if result.image is not None else frame

            # Show result
            cv2.imshow('ImagePRO - Live Hand Detection', display_frame)

            # Check for ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_hands_live()
