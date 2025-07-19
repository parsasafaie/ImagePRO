import mediapipe as mp
import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler


def analyze_face_mesh(max_faces=1, min_confidence=0.7, landmarks_idx=None, image_path=None, np_image=None, result_path=None):
    """
    Detects facial landmarks using MediaPipe FaceMesh and visualizes or saves them.

    Parameters:
        max_faces (int): Maximum number of faces to detect (default: 1).
        min_confidence (float): Minimum confidence threshold for detection (0.0 - 1.0).
        landmarks_idx (list): List of landmark indices to extract/visualize. If None, uses all 468.
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save output (image with landmarks or CSV). Supports `.jpg` and `.csv`.

    Returns:
        str | tuple(np.ndarray, list) | np.ndarray | list:
            - If `result_path` ends with `.jpg`: returns annotated image or saves it.
            - If `result_path` ends with `.csv`: returns list of coordinates or saves as CSV.
            - If no `result_path` is given: returns both annotated image and list of landmarks coordinates.

    Raises:
        ValueError: If inputs have invalid values or unsupported file extension.
    """
    # Validate specific parameters
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")

    if landmarks_idx is not None and not isinstance(landmarks_idx, list):
        raise TypeError("'landmarks_idx' must be a list of integers or None.")

    if result_path is not None and not isinstance(result_path, str):
        raise TypeError("'result_path' must be a string or None.")

    if result_path and not (result_path.endswith('.jpg') or result_path.endswith('.csv')):
        raise ValueError("Only '.jpg' and '.csv' extensions are supported for 'result_path'.")

    # Load input image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Initialize MediaPipe FaceMesh model
    mp_face_mesh = mp.solutions.face_mesh
    face_Mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=True
    )

    mp_drawing_utils = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Convert image to RGB for MediaPipe
    rgb_color = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    results = face_Mesh.process(rgb_color)

    if not results.multi_face_landmarks:
        raise ValueError("No face landmarks detected in the input image.")

    # Use default indices if none provided
    if landmarks_idx is None:
        landmarks_idx = list(range(468))

    # Process each detected face
    annotated_image = np_image.copy()
    all_landmarks = []

    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
        if result_path and result_path.endswith('.jpg') or result_path is None:
            if len(landmarks_idx) == 468:
                mp_drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            else:
                ih, iw, _ = annotated_image.shape
                for idx in landmarks_idx:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(iw * landmark.x), int(ih * landmark.y)
                    cv2.circle(annotated_image, (x, y), 3, (0, 0, 255), -1)

        if result_path and result_path.endswith('.csv') or result_path is None:
            landmarks_list = []
            for idx in landmarks_idx:
                landmark = face_landmarks.landmark[idx]
                landmarks_list.append([face_id, idx, landmark.x, landmark.y, landmark.z])
            all_landmarks.append(landmarks_list)

    # Handle output
    if result_path:
        if result_path.endswith('.jpg'):
            return IOHandler.save_image(annotated_image, result_path)
        elif result_path.endswith('.csv'):
            # Flatten list if multiple faces
            flat_landmarks = [item for sublist in all_landmarks for item in sublist]
            return IOHandler.save_csv(flat_landmarks, result_path)
    else:
        return annotated_image, all_landmarks
    

def analyze_face_mesh_live(max_faces=1, min_confidence=0.7):
    """
    Live webcam capture with real-time facial landmark detection and overlay.

    Parameters:
        max_faces (int): Maximum number of faces to detect.
        min_confidence (float): Minimum confidence threshold for face detection (0.0 - 1.0).

    Raises:
        ValueError: If inputs are out of valid range.
        RuntimeError: If camera cannot be accessed or released.
    """
    # Validate specific parameters
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    try:
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            try:
                landmarked_image = analyze_face_mesh(max_faces=max_faces, min_confidence=min_confidence, np_image=image)[0]
            except ValueError:
                landmarked_image = image

            cv2.imshow('ImagePRO - Live Face Mesh', landmarked_image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
