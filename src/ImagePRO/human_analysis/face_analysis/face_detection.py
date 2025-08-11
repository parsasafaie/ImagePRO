import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh


def detect_faces(
    max_faces=1,
    min_confidence=0.7,
    image_path=None,
    np_image=None,
    result_path=None,
    face_mesh_obj=None
):
    """
    Detect and crop face regions using MediaPipe face landmarks.

    Args:
        max_faces (int): Maximum number of faces to detect.
        min_confidence (float): Confidence threshold (0–1).
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Save path for cropped face(s).
        face_mesh_obj (FaceMesh, optional): Reusable MediaPipe instance.

    Returns:
        str or list[np.ndarray]: Confirmation message if saved, else list of cropped face images.

    Raises:
        TypeError, ValueError: On invalid inputs or if no faces detected.
    """
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be a float between 0 and 1.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    height, width = np_image.shape[:2]

    # Selected face outline landmark indices (based on MediaPipe's 468-point model)
    face_outline_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 164, 163, 153, 157
    ]

    # Run face mesh analysis
    _, raw_landmarks = analyze_face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=face_outline_indices,
        np_image=np_image,
        face_mesh_obj=face_mesh_obj
    )

    if not raw_landmarks:
        raise ValueError("No face landmarks detected in the input image.")

    # Convert normalized landmark coordinates to pixel positions
    all_polygons = []
    for face in raw_landmarks:
        polygon = [
            (int(x * width), int(y * height))
            for _, _, x, y, _ in face
        ]
        all_polygons.append(np.array(polygon, dtype=np.int32))

    # Crop faces using boundingRect
    cropped_faces = []
    for polygon in all_polygons:
        x, y, w, h = cv2.boundingRect(polygon)
        cropped = np_image[y:y + h, x:x + w]
        cropped_faces.append(cropped)

    return IOHandler.save_image(cropped_faces, result_path)
