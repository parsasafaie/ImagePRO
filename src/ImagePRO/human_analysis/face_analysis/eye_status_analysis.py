import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result
from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

# Constants
mp_face_mesh = mp.solutions.face_mesh
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_THRESHOLD = 0.2
RIGHT_EYE_INDICES = [386, 374, 263, 362]  # MediaPipe 468-point model


def analyze_eye_status(
    *,
    image: Image,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    face_mesh_obj=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Result:
    """
    Analyze right-eye open/closed status via Eye Aspect Ratio (EAR).

    Parameters
    ----------
    image : Image
        Image instance (BGR) to process.
    min_confidence : float, default=0.7
        Minimum detection confidence for FaceMesh in [0, 1].
    face_mesh_obj : mediapipe.python.solutions.face_mesh.FaceMesh | None, optional
        Reusable FaceMesh instance; if ``None``, one is created (static image mode).
    threshold : float, default=0.2
        EAR threshold below which the eye is considered closed.

    Returns
    -------
    Result
        `data` is a boolean: True if eye is open, False if closed. `image` is None. except when no face is detected or missing landmarks, then `data` is None and `meta` contains error info.

    Raises
    -------
    ValueError
        If inputs are invalid or no face landmarks detected or missing landmarks.
    """
    if not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    h, w = image.shape[:2]

    if face_mesh_obj is None:
        face_mesh_obj = mp_face_mesh.FaceMesh(
            min_detection_confidence=min_confidence,
            refine_landmarks=True,
            static_image_mode=True,
        )

    indices = RIGHT_EYE_INDICES

    mesh_result = analyze_face_mesh(
        max_faces=1,
        min_confidence=min_confidence,
        landmarks_idx=indices,
        image=image,
        face_mesh_obj=face_mesh_obj,
    )
    landmarks = mesh_result.data

    if not landmarks:
        return Result(image=None, data=None, meta={"source":image, "operation":"analyze_eye_status", "min_confidence": min_confidence, "threshold": threshold, "error": "No face landmarks detected"})

    eye_points = {lm[1]: lm for lm in landmarks[0]}

    try:
        top_y = eye_points[386][3] * h
        bottom_y = eye_points[374][3] * h
        left_x = eye_points[263][2] * w
        right_x = eye_points[362][2] * w
    except KeyError as e:
        return Result(image=None, data=None, meta={"source":image, "operation":"analyze_eye_status", "min_confidence": min_confidence, "threshold": threshold, "error": f"Missing landmark: {e}"})

    vertical_dist = abs(bottom_y - top_y)
    horizontal_dist = abs(right_x - left_x)

    is_open = False
    if horizontal_dist != 0:
        ear = vertical_dist / horizontal_dist
        is_open = ear > threshold

    return Result(image=None, data=is_open, meta={"source":image, "operation": "analyze_eye_status", "min_confidence": min_confidence, "threshold": threshold})


def analyze_eye_status_live(min_confidence: float = 0.7, threshold: float = 0.2) -> None:
    """
    Live eye open/closed detection using the default webcam.

    Parameters
    ----------
    min_confidence : float, default=0.7
        Minimum detection confidence in [0, 1].
    threshold : float, default=0.2
        EAR threshold to consider eyes open.
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
                img = Image.from_array(frame)
                result = analyze_eye_status(
                    image=img,
                    min_confidence=min_confidence,
                    face_mesh_obj=face_mesh,
                    threshold=threshold,
                )
                status = "Open" if result.data else "Closed"
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
