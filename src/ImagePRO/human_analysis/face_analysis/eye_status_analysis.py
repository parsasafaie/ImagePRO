from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[3]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh
from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

import cv2
import mediapipe as mp

# Constants
mp_face_mesh = mp.solutions.face_mesh
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_THRESHOLD = 0.2
RIGHT_EYE_INDICES = [386, 374, 263, 362]  # MediaPipe right eye landmarks


def analyze_eye_status(
    image: Image,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    threshold: float = DEFAULT_THRESHOLD,
    face_mesh_obj: mp.solutions.face_mesh.FaceMesh | None = None,
) -> Result:
    """Analyze if the right eye is open using Eye Aspect Ratio (EAR).

    Calculates eye openness by comparing vertical to horizontal distance
    between key eye landmarks. Uses FaceMesh landmark detection.

    Args:
        image: Input image to process.
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7
        threshold: EAR threshold for open vs closed.
            Eye considered open if EAR > threshold.
            Default: 0.2
        face_mesh_obj: Pre-initialized face mesh detector.
            If None, creates new instance in static mode.
            Default: None

    Returns:
        Result object with eye status:
        - data: True if eye is open, False if closed
            None if detection fails
        - meta: Operation info and error details if failed

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If min_confidence not in [0,1]
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(min_confidence, (int, float)):
        raise TypeError("'min_confidence' must be a number")
    if not 0 <= min_confidence <= 1:
        raise ValueError("'min_confidence' must be between 0 and 1")

    # Get image dimensions
    h, w = image.shape[:2]

    # Initialize detector if needed
    if face_mesh_obj is None:
        face_mesh_obj = mp_face_mesh.FaceMesh(
            min_detection_confidence=min_confidence,
            refine_landmarks=True,
            static_image_mode=True
        )

    # Get face landmarks
    mesh_result = analyze_face_mesh(
        image=image,
        max_faces=1,
        min_confidence=min_confidence,
        landmarks_idx=RIGHT_EYE_INDICES,
        face_mesh_obj=face_mesh_obj
    )
    landmarks = mesh_result.data

    # Handle no detections
    if not landmarks:
        return Result(
            image=None,
            data=None,
            meta={
                "source": image,
                "operation": "analyze_eye_status",
                "min_confidence": min_confidence,
                "threshold": threshold,
                "error": "No face landmarks detected"
            }
        )

    # Map landmarks by index
    eye_points = {lm[1]: lm for lm in landmarks[0]}

    try:
        # Extract key points (scale to image dimensions)
        top_y = eye_points[386][3] * h       # Top of eye
        bottom_y = eye_points[374][3] * h    # Bottom of eye
        left_x = eye_points[263][2] * w      # Left corner
        right_x = eye_points[362][2] * w     # Right corner
    except KeyError as e:
        return Result(
            image=None,
            data=None,
            meta={
                "source": image,
                "operation": "analyze_eye_status",
                "min_confidence": min_confidence,
                "threshold": threshold,
                "error": f"Missing landmark: {e}"
            }
        )

    # Calculate Eye Aspect Ratio (EAR)
    vertical_dist = abs(bottom_y - top_y)
    horizontal_dist = abs(right_x - left_x)

    # Determine eye state
    is_open = False
    if horizontal_dist > 0:  # Avoid division by zero
        ear = vertical_dist / horizontal_dist
        is_open = ear > threshold

    return Result(
        image=None,
        data=is_open,
        meta={
            "source": image,
            "operation": "analyze_eye_status",
            "min_confidence": min_confidence,
            "threshold": threshold
        }
    )


def analyze_eye_status_live(
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    threshold: float = DEFAULT_THRESHOLD
) -> None:
    """Run live eye status detection using webcam feed.

    Opens a window displaying webcam feed with overlaid eye status.
    Press ESC to exit.

    Args:
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7
        threshold: EAR threshold for open vs closed.
            Eye considered open if EAR > threshold.
            Default: 0.2

    Raises:
        TypeError: If min_confidence is not a number
        ValueError: If min_confidence not in [0,1]
        RuntimeError: If webcam cannot be accessed
    """
    # Validate inputs
    if not isinstance(min_confidence, (int, float)):
        raise TypeError("'min_confidence' must be a number")
    if not 0 <= min_confidence <= 1:
        raise ValueError("'min_confidence' must be between 0 and 1")

    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Failed to access webcam")

    # Initialize face mesh detector for video
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=False  # Optimize for video
    )

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame, skipping...")
                continue

            # Process frame
            try:
                img = Image.from_array(frame)
                result = analyze_eye_status(
                    image=img,
                    min_confidence=min_confidence,
                    face_mesh_obj=face_mesh,
                    threshold=threshold
                )
                status = "Open" if result.data else "Closed"
            except (TypeError, ValueError):
                status = "No face"

            # Draw results
            color = (0, 255, 0) if status == "Open" else (0, 0, 255)
            cv2.putText(
                frame,
                f"Eye: {status}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Display frame
            cv2.imshow("ImagePRO - Eye Status", frame)

            # Check for ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_eye_status_live()
