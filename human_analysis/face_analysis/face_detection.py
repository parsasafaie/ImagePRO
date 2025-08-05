import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh


def detect_faces(max_faces=1, min_confidence=0.7, image_path=None, np_image=None, result_path=None, face_mesh_obj=None):
    """
    Extracts face regions from an input image using detected facial landmarks.

    Parameters:
        max_faces (int): Maximum number of faces to detect (default: 1).
        min_confidence (float): Minimum confidence threshold for detection (0.0 - 1.0).
        image_path (str | None): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray | None): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str | None): Path to save the cropped face(s). Supports `.jpg` and `.csv`.
        face_mesh_obj (mp.solutions.face_mesh.FaceMesh): Optional external FaceMesh instance.
            If provided, this instance will be used instead of creating a new one. Useful for real-time/live use cases to avoid repeated model creation.

    Returns:
        str | list[np.ndarray]: 
            - If `result_path` is given, returns confirmation message.
            - Otherwise, returns list of cropped face images.

    Raises:
        TypeError: If inputs are of incorrect type.
        ValueError: If values are out of valid range or no face is detected.
    """
    # Validate specific parameters
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    h, w, _ = np_image.shape

    # Define face outline indices
    face_outline_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 164, 163, 153, 157
    ]

    # Get annotated landmarks
    landmarks = analyze_face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=face_outline_indices,
        np_image=np_image,
        face_mesh_obj=face_mesh_obj
    )[1]

    if not landmarks:
        raise ValueError("No face landmarks detected in the input image.")

    # Convert normalized landmarks to pixel coordinates
    adjusted_landmarks = []
    for face_index, face in enumerate(landmarks):
        adjusted_face = []
        for landmark_index, landmark in enumerate(face):
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            adjusted_face.append([x, y])
        adjusted_landmarks.append(np.array(adjusted_face, dtype=np.int32))

    # Crop each face region
    cropped_faces = []
    for face in adjusted_landmarks:
        x, y, w_rect, h_rect = cv2.boundingRect(face)
        cropped_face = np_image[y:y+h_rect, x:x+w_rect]
        cropped_faces.append(cropped_face)

    # Save or return result
    return IOHandler.save_image(cropped_faces, result_path=result_path)