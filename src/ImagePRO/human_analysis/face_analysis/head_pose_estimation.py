import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result
from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

# Constants
mp_face_mesh = mp.solutions.face_mesh
DEFAULT_MAX_FACES = 1
DEFAULT_MIN_CONFIDENCE = 0.7
HEAD_POSE_INDICES = [1, 152, 33, 263, 168]  # nose_tip, chin, left_eye, right_eye, nasion


def estimate_head_pose(
    *,
    image: Image,
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    face_mesh_obj=None,
) -> Result:
    """
    Estimate head pose (yaw, pitch) from a single image using MediaPipe facial landmarks.

    Parameters
    ----------
    image : Image
        Image instance (BGR) to process.
    max_faces : int, default=1
        Number of faces to detect.
    min_confidence : float, default=0.7
        Detection confidence threshold in [0, 1].
    face_mesh_obj : mediapipe.python.solutions.face_mesh.FaceMesh | None, optional
        Optional reusable FaceMesh instance.

    Returns
    -------
    Result
        `data` is a list of [face_id, yaw, pitch] per face; `image` is None. except if no face is detected or missing landmarks, then `data` is None and `meta` contains error info.

    Raises
    -------
    ValueError
        If inputs are invalid or no face landmarks detected or missing landmarks.
    """
    if not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")


    indices = HEAD_POSE_INDICES

    mesh_result = analyze_face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=indices,
        image=image,
        face_mesh_obj=face_mesh_obj,
    )
    landmarks = mesh_result.data

    if not landmarks:
        return Result(image=None, data=None, meta={"source":image, "operation": "estimate_head_pose", "max_faces": max_faces, "min_confidence": min_confidence, "error": "No face landmarks detected."})

    results = []
    for face in landmarks:
        points = {lm[1]: lm for lm in face}
        try:
            nose_x, nose_y = points[1][2:4]
            chin_y = points[152][3]
            left_x = points[33][2]
            right_x = points[263][2]
            nasion_x, nasion_y = points[168][2:4]
        except KeyError:
            return Result(image=None, data=None, meta={"source":image, "operation": "estimate_head_pose", "max_faces": max_faces, "min_confidence": min_confidence, "error": "Missing necessary landmarks."})


        yaw = 100 * ((right_x - nasion_x) - (nasion_x - left_x))
        pitch = 100 * ((chin_y - nose_y) - (nose_y - nasion_y))
        results.append([face[0][0], yaw, pitch])

    return Result(image=None, data=results, meta={"source":image, "operation": "estimate_head_pose", "max_faces": max_faces, "min_confidence": min_confidence})


def estimate_head_pose_live(max_faces: int = 1, min_confidence: float = 0.7):
    """
    Live head pose estimation using the default webcam.

    Parameters
    ----------
    max_faces : int, default=1
        Number of faces to detect per frame.
    min_confidence : float, default=0.7
        Detection confidence threshold in [0, 1].
    """
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Failed to access webcam.")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=False
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Skipping empty frame.")
                continue

            try:
                img = Image.from_array(frame)
                result = estimate_head_pose(
                    image=img,
                    max_faces=max_faces,
                    min_confidence=min_confidence,
                    face_mesh_obj=face_mesh,
                )
                face_angles = result.data or []
            except ValueError:
                face_angles = []

            for i, face in enumerate(face_angles):
                face_id, yaw, pitch = face
                text = f"Face {int(face_id)+1}: Yaw={yaw:.2f}, Pitch={pitch:.2f}"
                cv2.putText(frame, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("ImagePRO - Head Pose Estimation", frame)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    estimate_head_pose_live()
