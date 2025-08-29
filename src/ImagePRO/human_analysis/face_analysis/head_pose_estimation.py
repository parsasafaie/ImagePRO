import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

# Constants
mp_face_mesh = mp.solutions.face_mesh
DEFAULT_MAX_FACES = 1
DEFAULT_MIN_CONFIDENCE = 0.7
HEAD_POSE_INDICES = [1, 152, 33, 263, 168]  # nose_tip, chin, left_eye, right_eye, nasion


def estimate_head_pose(
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    src_image_path: str | None = None,
    src_np_image=None,
    output_csv_path: str | None = None,
    face_mesh_obj=None,
):
    """
    Estimate head pose (yaw, pitch) from a single image using MediaPipe facial landmarks.

    Parameters
    ----------
    max_faces : int, default=1
        Number of faces to detect.
    min_confidence : float, default=0.7
        Detection confidence threshold in [0, 1].
    src_image_path : str | None, optional
        Path to the input image file.
    src_np_image : np.ndarray | None, optional
        Input image array in BGR format. If both are provided, `src_np_image` takes precedence.
    output_csv_path : str | None, optional
        If provided, saves the results as CSV rows: [face_id, yaw, pitch].
    face_mesh_obj : mediapipe.python.solutions.face_mesh.FaceMesh | None, optional
        Optional reusable FaceMesh instance.

    Returns
    -------
    list[list[float]] | str
        Save message from IOHandler.save_csv if `output_csv_path` is provided,
        otherwise list of [face_id, yaw, pitch] for each detected face.

    Raises
    ------
    ValueError
        * If inputs are invalid or no face landmarks are detected (this function).
        * From IOHandler.load_image when both inputs are None or image loading fails.
    TypeError
        From IOHandler.load_image/save_csv on invalid argument types.
    FileNotFoundError
        From IOHandler.load_image when `src_image_path` does not exist.
    """
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)

    # Important landmark indices (MediaPipe 468 model)
    indices = HEAD_POSE_INDICES

    _, landmarks = analyze_face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=indices,
        src_np_image=np_image,
        face_mesh_obj=face_mesh_obj
    )

    if not landmarks:
        raise ValueError("No face landmarks detected.")

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
            continue  # skip this face if any point is missing

        # Simplified proportional estimation on normalized coords
        yaw = 100 * ((right_x - nasion_x) - (nasion_x - left_x))
        pitch = 100 * ((chin_y - nose_y) - (nose_y - nasion_y))

        results.append([face[0][0], yaw, pitch])

    if output_csv_path:
        print(IOHandler.save_csv(results, output_csv_path))

    return results


def estimate_head_pose_live(max_faces: int = 1, min_confidence: float = 0.7):
    """
    Live head pose estimation using the default webcam.

    Parameters
    ----------
    max_faces : int, default=1
        Number of faces to detect per frame.
    min_confidence : float, default=0.7
        Detection confidence threshold in [0, 1].

    Raises
    ------
    ValueError
        On invalid inputs.
    RuntimeError
        If the camera cannot be opened.

    Notes
    -----
    * Uses static_image_mode=False for better performance on video streams.
    * Press ESC to exit the preview window.
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
                face_angles = estimate_head_pose(
                    max_faces=max_faces,
                    min_confidence=min_confidence,
                    src_np_image=frame,
                    face_mesh_obj=face_mesh
                )
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
