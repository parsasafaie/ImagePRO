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
DEFAULT_MAX_FACES = 1
DEFAULT_MIN_CONFIDENCE = 0.7
HEAD_POSE_INDICES = [1, 152, 33, 263, 168]  # nose_tip, chin, left_eye, right_eye, nasion


def estimate_head_pose(
    image: Image,
    *,
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    face_mesh_obj: mp.solutions.face_mesh.FaceMesh | None = None
) -> Result:
    """Estimate head pose angles using facial landmarks.

    Calculates approximate yaw and pitch angles based on relative
    positions of key facial landmarks (nose, eyes, chin).

    Args:
        image: Input image to process.
        max_faces: Maximum number of faces to analyze.
            Must be positive.
            Default: 1
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7
        face_mesh_obj: Pre-initialized face mesh detector.
            If None, creates new instance.
            Default: None

    Returns:
        Result object with pose estimates:
        - data: List of [face_id, yaw, pitch] per face
            None if no faces detected
        - meta: Operation info and parameters
            Includes error info if detection fails

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If max_faces is not positive
        ValueError: If min_confidence not in [0,1]
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be positive")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1")

    # Get face landmarks
    mesh_result = analyze_face_mesh(
        image=image,
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=HEAD_POSE_INDICES,
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
                "operation": "estimate_head_pose",
                "max_faces": max_faces,
                "min_confidence": min_confidence,
                "error": "No face landmarks detected"
            }
        )

    # Calculate angles for each face
    pose_data = []
    for face in landmarks:
        # Map landmarks by index
        points = {lm[1]: lm for lm in face}
        
        try:
            # Extract key points
            nose_x, nose_y = points[1][2:4]      # Nose tip
            chin_y = points[152][3]              # Chin
            left_x = points[33][2]               # Left eye
            right_x = points[263][2]             # Right eye
            nasion_x, nasion_y = points[168][2:4] # Nose bridge
        except KeyError:
            return Result(
                image=None,
                data=None,
                meta={
                    "source": image,
                    "operation": "estimate_head_pose",
                    "max_faces": max_faces,
                    "min_confidence": min_confidence,
                    "error": "Missing required landmarks"
                }
            )

        # Calculate angles
        yaw = 100 * ((right_x - nasion_x) - (nasion_x - left_x))    # Horizontal rotation
        pitch = 100 * ((chin_y - nose_y) - (nose_y - nasion_y))     # Vertical rotation
        pose_data.append([face[0][0], yaw, pitch])

    return Result(
        image=None,
        data=pose_data,
        meta={
            "source": image,
            "operation": "estimate_head_pose",
            "max_faces": max_faces,
            "min_confidence": min_confidence
        }
    )


def estimate_head_pose_live(
    *,
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> None:
    """Run live head pose estimation using webcam feed.

    Opens a window displaying webcam feed with overlaid head pose angles.
    Press ESC to exit.

    Args:
        max_faces: Maximum number of faces to analyze per frame.
            Must be positive.
            Default: 1
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7

    Raises:
        TypeError: If max_faces is not an integer
        ValueError: If max_faces is not positive
        ValueError: If min_confidence not in [0,1]
        RuntimeError: If webcam cannot be accessed
    """
    # Validate inputs
    if not isinstance(max_faces, int):
        raise TypeError("'max_faces' must be an integer")
    if max_faces <= 0:
        raise ValueError("'max_faces' must be positive")
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
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=False
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
                result = estimate_head_pose(
                    image=img,
                    max_faces=max_faces,
                    min_confidence=min_confidence,
                    face_mesh_obj=face_mesh
                )
                face_angles = result.data or []
            except (TypeError, ValueError):
                face_angles = []

            # Draw results
            for i, face in enumerate(face_angles):
                face_id, yaw, pitch = face
                text = f"Face {int(face_id)+1}: Yaw={yaw:.1f}, Pitch={pitch:.1f}"
                cv2.putText(
                    frame, text,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

            # Display frame
            cv2.imshow("ImagePRO - Head Pose Estimation", frame)

            # Check for ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    estimate_head_pose_live()
