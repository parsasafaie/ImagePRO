import mediapipe as mp
import cv2
import numpy as np
from pathlib import Path

# Import new IOHandler
from io_handler import IOHandler


def face_mesh(max_faces=1, min_confidence=0.7, landmarks_idx=None, image_path=None, np_image=None, result_path=None):
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
        str | tuple(np.ndarray | list): 
            - If `result_path` ends with `.jpg`, returns annotated image or saves it.
            - If `result_path` ends with `.csv`, returns list of coordinates or saves as CSV.
            - If no `result_path` is given, returns the annotated image or list of coordinates.

    Raises:
        TypeError: If inputs are of incorrect type.
        ValueError: If values are out of valid range or unsupported file extension.
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
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=True
    )

    # Convert image to RGB for MediaPipe
    rgb_color = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_color)

    if not results.multi_face_landmarks:
        raise ValueError("No face landmarks detected in the input image.")

    # Use default indices if none provided
    if landmarks_idx is None:
        landmarks_idx = list(range(468))

    # Process each detected face
    all_landmarks = []

    for face_landmarks in results.multi_face_landmarks:
        if result_path and result_path.endswith('.jpg'):
            ih, iw, _ = np_image.shape
            for idx in landmarks_idx:
                landmark = face_landmarks.landmark[idx]
                x, y = int(iw * landmark.x), int(ih * landmark.y)
                cv2.circle(np_image, (x, y), 1, (0, 255, 0), -1)

        elif result_path and result_path.endswith('.csv'):
            landmarks_list = []
            for idx in landmarks_idx:
                landmark = face_landmarks.landmark[idx]
                landmarks_list.append([landmark.x, landmark.y, landmark.z])
            all_landmarks.extend(landmarks_list)

    # Handle output
    if result_path:
        if result_path.endswith('.jpg'):
            return IOHandler.save_image(np_image, result_path)
        elif result_path.endswith('.csv'):
            return IOHandler.save_csv(all_landmarks, result_path)
    else:
        if result_path is None and landmarks_idx is None:
            return np_image
        elif result_path is None and landmarks_idx is not None:
            return all_landmarks
        