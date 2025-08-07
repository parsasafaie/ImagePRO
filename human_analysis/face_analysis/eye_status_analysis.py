import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

mp_face_mesh = mp.solutions.face_mesh


def analyze_eye_status(
    min_confidence=0.7,
    image_path=None,
    np_image=None,
    face_mesh_obj=None,
    threshold=0.2
):
    """
    Analyze eye open/closed status using Eye Aspect Ratio (EAR).

    Args:
        min_confidence (float): Minimum confidence for FaceMesh detection.
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        face_mesh_obj (FaceMesh, optional): Reusable instance of FaceMesh.
        threshold (float): EAR threshold below which eye is considered closed.

    Returns:
        bool: True if eye is open, False if closed.

    Raises:
        ValueError: If landmarks are not detected or inputs are invalid.
    """
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    h, w = np_image.shape[:2]

    if face_mesh_obj is None:
        face_mesh_obj = mp_face_mesh.FaceMesh(
            min_detection_confidence=min_confidence,
            refine_landmarks=True,
            static_image_mode=True
        )

    # Landmark indices for right eye (from MediaPipe's model)
    # 386 (top eyelid), 374 (bottom eyelid), 263 (outer corner), 362 (inner corner)
    indices = [386, 374, 263, 362]

    _, landmarks = analyze_face_mesh(
        max_faces=1,
        min_confidence=min_confidence,
        landmarks_idx=indices,
        np_image=np_image,
        face_mesh_obj=face_mesh_obj
    )

    if not landmarks:
        raise ValueError("No face landmarks detected.")

    eye_points = {lm[1]: lm for lm in landmarks[0]}

    try:
        top_y = eye_points[386][3] * h
        bottom_y = eye_points[374][3] * h
        left_x = eye_points[263][2] * w
        right_x = eye_points[362][2] * w
    except KeyError:
        raise ValueError("Missing necessary eye landmarks.")

    vertical_dist = abs(bottom_y - top_y)
    horizontal_dist = abs(right_x - left_x)

    if horizontal_dist == 0:
        return False  # avoid division by zero

    ear = vertical_dist / horizontal_dist
    return ear > threshold


def analyze_eye_status_live(min_confidence=0.7, threshold=0.2):
    """
    Perform live eye open/closed detection using webcam input.

    Args:
        min_confidence (float): Minimum confidence for detection.
        threshold (float): EAR threshold to consider eyes open.

    Raises:
        ValueError: If confidence value is invalid.
        RuntimeError: If webcam cannot be accessed.
    """
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam.")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=True
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Skipping empty frame.")
                continue

            try:
                is_open = analyze_eye_status(
                    min_confidence=min_confidence,
                    np_image=frame,
                    face_mesh_obj=face_mesh,
                    threshold=threshold
                )
                status = "Open" if is_open else "Closed"
            except ValueError:
                status = "No face"

            cv2.putText(
                frame,
                f"Eye: {status}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if status == "Open" else (0, 0, 255),
                2
            )

            cv2.imshow("ImagePRO - Eye Status (ESC to Exit)", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_eye_status_live()
