import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

# Constants
mp_face_mesh = mp.solutions.face_mesh
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_THRESHOLD = 0.2
RIGHT_EYE_INDICES = [386, 374, 263, 362]  # MediaPipe 468-point model


def analyze_eye_status(
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    src_image_path: str | None = None,
    src_np_image=None,
    face_mesh_obj=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """
    Analyze right-eye open/closed status via Eye Aspect Ratio (EAR).

    Parameters
    ----------
    min_confidence : float, default=0.7
        Minimum detection confidence for FaceMesh in [0, 1].
    src_image_path : str | None, optional
        Path to input image file.
    src_np_image : np.ndarray | None, optional
        Image array in BGR (OpenCV). If both provided, ``src_np_image`` takes precedence.
    face_mesh_obj : mediapipe.python.solutions.face_mesh.FaceMesh | None, optional
        Reusable FaceMesh instance; if ``None``, one is created (static image mode).
    threshold : float, default=0.2
        EAR threshold below which the eye is considered closed.

    Returns
    -------
    bool
        ``True`` if eye is open, ``False`` if closed.

    Raises
    ------
    ValueError
        * If ``min_confidence`` is out of range, or landmarks are not detected.
        * From ``IOHandler.load_image`` when both inputs are ``None`` or image loading fails.
    TypeError
        From ``IOHandler.load_image`` on invalid argument types.
    FileNotFoundError
        From ``IOHandler.load_image`` when ``src_image_path`` does not exist.

    Notes
    -----
    * Uses MediaPipe indices for the right eye:
      386 (upper lid), 374 (lower lid), 263 (outer corner), 362 (inner corner).
    * EAR is computed as ``vertical / horizontal`` on pixel coordinates.
    """
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    h, w = np_image.shape[:2]

    if face_mesh_obj is None:
        face_mesh_obj = mp_face_mesh.FaceMesh(
            min_detection_confidence=min_confidence,
            refine_landmarks=True,
            static_image_mode=True,
        )

    # Landmark indices for right eye (MediaPipe 468-point model)
    indices = RIGHT_EYE_INDICES

    _, landmarks = analyze_face_mesh(
        max_faces=1,
        min_confidence=min_confidence,
        landmarks_idx=indices,
        src_np_image=np_image,
        face_mesh_obj=face_mesh_obj,
    )

    if not landmarks:
        raise ValueError("No face landmarks detected.")

    eye_points = {lm[1]: lm for lm in landmarks[0]}

    try:
        top_y = eye_points[386][3] * h
        bottom_y = eye_points[374][3] * h
        left_x = eye_points[263][2] * w
        right_x = eye_points[362][2] * w
    except KeyError as e:
        raise ValueError("Missing necessary eye landmarks.") from e

    vertical_dist = abs(bottom_y - top_y)
    horizontal_dist = abs(right_x - left_x)

    if horizontal_dist == 0:
        return False  # avoid division by zero

    ear = vertical_dist / horizontal_dist
    return ear > threshold


def analyze_eye_status_live(min_confidence: float = 0.7, threshold: float = 0.2) -> None:
    """
    Live eye open/closed detection using the default webcam.

    Parameters
    ----------
    min_confidence : float, default=0.7
        Minimum detection confidence in [0, 1].
    threshold : float, default=0.2
        EAR threshold to consider eyes open.

    Raises
    ------
    ValueError
        If ``min_confidence`` is invalid.
    RuntimeError
        If the webcam cannot be accessed.

    Notes
    -----
    * Uses ``static_image_mode=False`` for better performance on video streams.
    * Press **ESC** to exit.
    """
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam.")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=False,  # better for live video
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
                    src_np_image=frame,
                    face_mesh_obj=face_mesh,
                    threshold=threshold,
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
                2,
            )

            cv2.imshow("ImagePRO - Eye Status (ESC to Exit)", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_eye_status_live()
