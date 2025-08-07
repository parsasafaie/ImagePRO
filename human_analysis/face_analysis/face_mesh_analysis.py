import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler

mp_face_mesh = mp.solutions.face_mesh
mp_drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def analyze_face_mesh(
    max_faces=1,
    min_confidence=0.7,
    landmarks_idx=None,
    image_path=None,
    np_image=None,
    result_path=None,
    face_mesh_obj=None
):
    """
    Detect facial landmarks using MediaPipe FaceMesh.

    Args:
        max_faces (int): Maximum number of faces to detect.
        min_confidence (float): Detection confidence threshold.
        landmarks_idx (list, optional): Indices of landmarks to extract or draw.
        image_path (str, optional): Path to input image file.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Save path (supports .jpg or .csv).
        face_mesh_obj (FaceMesh, optional): External instance to reuse.

    Returns:
        str or tuple or np.ndarray or list:
            - If `result_path` ends with `.jpg`: save/return annotated image.
            - If `result_path` ends with `.csv`: save/return coordinates.
            - If no result_path: returns (image, landmarks list).
    """
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be a float between 0 and 1.")

    if landmarks_idx is not None and not isinstance(landmarks_idx, list):
        raise TypeError("'landmarks_idx' must be a list of integers or None.")

    if result_path and not isinstance(result_path, str):
        raise TypeError("'result_path' must be a string or None.")

    if result_path and not result_path.endswith(('.jpg', '.csv')):
        raise ValueError("Only '.jpg' or '.csv' formats are supported.")

    if face_mesh_obj is None:
        face_Mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=min_confidence,
            refine_landmarks=True,
            static_image_mode=True
        )
    else:
        face_Mesh = face_mesh_obj

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    results = face_Mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        raise ValueError("No face landmarks detected.")

    landmarks_idx = landmarks_idx or list(range(468))
    annotated = np_image.copy()
    all_landmarks = []

    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
        if result_path is None or result_path.endswith('.jpg'):
            if len(landmarks_idx) == 468:
                mp_drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            else:
                h, w = annotated.shape[:2]
                for idx in landmarks_idx:
                    lm = face_landmarks.landmark[idx]
                    cx, cy = int(w * lm.x), int(h * lm.y)
                    cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

        if result_path is None or result_path.endswith('.csv'):
            face_data = [
                [face_id, idx, lm.x, lm.y, lm.z]
                for idx in landmarks_idx
                for lm in [face_landmarks.landmark[idx]]
            ]
            all_landmarks.append(face_data)

    # Output handling
    if result_path:
        if result_path.endswith('.jpg'):
            return IOHandler.save_image(annotated, result_path)
        elif result_path.endswith('.csv'):
            flat_data = [row for face in all_landmarks for row in face]
            return IOHandler.save_csv(flat_data, result_path)

    return annotated, all_landmarks


def analyze_face_mesh_live(max_faces=1, min_confidence=0.7):
    """
    Launch webcam with real-time face landmark overlay.

    Args:
        max_faces (int): Max faces to detect.
        min_confidence (float): Detection confidence threshold.

    Raises:
        RuntimeError: If webcam can't be opened.
    """
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    face_Mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
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
                result_image, _ = analyze_face_mesh(
                    max_faces=max_faces,
                    min_confidence=min_confidence,
                    np_image=frame,
                    face_mesh_obj=face_Mesh
                )
            except ValueError:
                result_image = frame

            cv2.imshow("ImagePRO - Face Mesh", result_image)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_face_mesh_live(max_faces=1)
