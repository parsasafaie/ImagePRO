import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

# Constants
DEFAULT_MAX_FACES = 1
DEFAULT_MIN_CONFIDENCE = 0.7
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 164, 163, 153, 157
]


def detect_faces(
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None,
    face_mesh_obj=None,
):
    """
    Detect and crop face regions using MediaPipe facial landmarks.

    Parameters
    ----------
    max_faces : int, default=1
        Maximum number of faces to detect.
    min_confidence : float, default=0.7
        Detection confidence threshold in [0, 1].
    src_image_path : str | None, optional
        Path to input image file.
    src_np_image : np.ndarray | None, optional
        Image array in BGR format. If both provided, `src_np_image` takes precedence.
    output_image_path : str | None, optional
        If provided, saves cropped faces to disk (single path used as base name for multiple faces).
    face_mesh_obj : mediapipe.python.solutions.face_mesh.FaceMesh | None, optional
        Optional reusable FaceMesh instance.

    Returns
    -------
    list[np.ndarray] | str
        If `output_image_path` is provided, returns the save message from
        `IOHandler.save_image`. Otherwise, returns a list of cropped face images.

    Raises
    ------
    ValueError
        * If inputs are invalid or no face landmarks are detected (this function).
        * From IOHandler.load_image when both inputs are None or image loading fails.
    TypeError
        From IOHandler.load_image/save_image on invalid argument types.
    FileNotFoundError
        From IOHandler.load_image when `src_image_path` does not exist.
    IOError
        From IOHandler.save_image when saving the image fails.
    """
    # --- Validate inputs -----------------------------------------------------
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be a float between 0 and 1.")

    # --- Load image -----------------------------------------------------------
    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    height, width = np_image.shape[:2]

    # Selected face outline landmark indices (MediaPipe 468-point model)
    face_outline_indices = FACE_OUTLINE_INDICES

    # --- Run face mesh analysis ----------------------------------------------
    _, raw_landmarks = analyze_face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=face_outline_indices,
        src_np_image=np_image,
        face_mesh_obj=face_mesh_obj
    )

    if not raw_landmarks:
        raise ValueError("No face landmarks detected in the input image.")

    # --- Convert normalized landmark coordinates to pixel positions ----------
    all_polygons = []
    for face in raw_landmarks:
        polygon = [
            (int(x * width), int(y * height))
            for _, _, x, y, _ in face
        ]
        all_polygons.append(np.array(polygon, dtype=np.int32))

    # --- Crop faces using boundingRect ---------------------------------------
    cropped_faces = []
    for polygon in all_polygons:
        x, y, w, h = cv2.boundingRect(polygon)
        cropped = np_image[y:y + h, x:x + w]
        cropped_faces.append(cropped)

    # --- Output handling ------------------------------------------------------
    if output_image_path:
        print(IOHandler.save_image(cropped_faces, output_image_path))

    return cropped_faces
