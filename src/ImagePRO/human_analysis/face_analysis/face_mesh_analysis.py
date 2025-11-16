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
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
DEFAULT_MAX_FACES = 1
DEFAULT_MIN_CONFIDENCE = 0.7
TOTAL_FACE_LANDMARKS = 468


def analyze_face_mesh(
    image: Image,
    *,
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    landmarks_idx: list[int] | None = None,
    face_mesh_obj: mp.solutions.face_mesh.FaceMesh | None = None
) -> Result:
    """Detect facial landmarks using MediaPipe FaceMesh.

    Extracts 3D facial mesh coordinates for precise face analysis.
    Can focus on specific landmarks or detect full 468-point mesh.

    Args:
        image: Input image to process.
        max_faces: Maximum number of faces to detect.
            Must be positive.
            Default: 1
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7
        landmarks_idx: Specific landmark indices to detect.
            If None, uses all 468 points.
            Default: None
        face_mesh_obj: Pre-initialized face mesh detector.
            If None, creates new instance.
            Default: None

    Returns:
        Result object with detections and visualization:
        - image: Input image with landmarks drawn
            None if no faces detected
        - data: List of [face_id, point_id, x, y, z]
            None if no faces detected
        - meta: Operation info and parameters
            Includes error info if detection fails

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If max_faces is not positive
        ValueError: If min_confidence not in [0,1]
        TypeError: If landmarks_idx is not list[int]

    Notes:
        - Coordinates are normalized [0,1]. Multiply by width/height for pixels
        - Full mesh (468 points) shows tessellation, specific points show dots
    """

    # Validate inputs
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be positive")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1")

    if landmarks_idx is not None and (
        not isinstance(landmarks_idx, list) 
        or not all(isinstance(i, int) for i in landmarks_idx)
    ):
        raise TypeError("'landmarks_idx' must be a list of integers")

    # Initialize detector if needed
    if face_mesh_obj is None:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=min_confidence,
            refine_landmarks=True,
            static_image_mode=True
        )
    else:
        face_mesh = face_mesh_obj

    # Prepare image
    img_copy = image._data.copy()
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    
    # Detect facial landmarks
    results = face_mesh.process(img_rgb)

    # Handle no detections
    if not results.multi_face_landmarks:
        return Result(
            image=None,
            data=None,
            meta={
                "source": image,
                "operation": "analyze_face_mesh",
                "landmarks_idx": landmarks_idx,
                "max_faces": max_faces,
                "min_confidence": min_confidence,
                "error": "No face landmarks detected"
            }
        )

    # Use all landmarks if none specified
    landmarks_idx = landmarks_idx or list(range(TOTAL_FACE_LANDMARKS))
    # Process each detected face
    landmarks = []
    
    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
        # Draw landmarks
        if len(landmarks_idx) == TOTAL_FACE_LANDMARKS:
            # Draw full face mesh
            mp_drawing_utils.draw_landmarks(
                image=img_copy,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        else:
            # Draw specific points
            h, w = img_copy.shape[:2]
            for idx in landmarks_idx:
                lm = face_landmarks.landmark[idx]
                cx, cy = int(w * lm.x), int(h * lm.y)
                cv2.circle(img_copy, (cx, cy), 3, (0, 0, 255), -1)

        # Extract coordinates
        face_data = [
            [face_id, idx, lm.x, lm.y, lm.z]
            for idx in landmarks_idx
            for lm in [face_landmarks.landmark[idx]]
        ]
        landmarks.append(face_data)

    return Result(
        image=img_copy,
        data=landmarks,
        meta={
            "source": image,
            "operation": "analyze_face_mesh",
            "landmarks_idx": landmarks_idx,
            "max_faces": max_faces,
            "min_confidence": min_confidence
        }
    )


def analyze_face_mesh_live(
    *,
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> None:
    """Start live webcam feed with real-time face mesh visualization.

    Opens default camera and shows face landmarks in real-time.
    Uses tracking mode for better performance on video.
    Press ESC to exit.

    Args:
        max_faces: Maximum number of faces to track.
            Must be positive.
            Default: 1
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If camera cannot be accessed
    """
    # Validate parameters
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be positive")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1")

    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam")

    # Initialize detector in tracking mode for better performance
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
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
                result = analyze_face_mesh(
                    image=Image.from_array(frame),
                    max_faces=max_faces,
                    min_confidence=min_confidence,
                    face_mesh_obj=face_mesh
                )
                display_frame = result.image if result.image is not None else frame
            except (TypeError, ValueError):
                # Fall back to raw frame if detection fails
                display_frame = frame

            # Show result
            cv2.imshow("ImagePRO - Face Mesh", display_frame)

            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_face_mesh_live()
