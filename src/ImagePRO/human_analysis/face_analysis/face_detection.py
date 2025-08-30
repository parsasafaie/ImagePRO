from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh
from ImagePRO.utils.result import Result
from ImagePRO.utils.image import Image
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))


# Constants
DEFAULT_MAX_FACES = 1
DEFAULT_MIN_CONFIDENCE = 0.7
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 164, 163, 153, 157
]


def detect_faces(
    *,
    image: Image, 
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    face_mesh_obj=None,
) -> Result:
    """
    Detect and crop face regions using MediaPipe facial landmarks.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to process.
    max_faces : int, default=1
        Maximum number of faces to detect.
    min_confidence : float, default=0.7
        Detection confidence threshold in [0, 1].
    face_mesh_obj : mediapipe.python.solutions.face_mesh.FaceMesh | None, optional
        Optional reusable FaceMesh instance.

    Returns
    -------
    Result
        Result where `image` may be a list[np.ndarray] of cropped faces.
        `data` contains landmark points per face in pixel coordinates. 
        except if no face is detected or missing landmarks, then `image` is None and `data` is None and meta has error info.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be a float between 0 and 1.")

    np_image = image._data
    height, width = np_image.shape[:2]

    # Selected face outline indices
    face_outline_indices = FACE_OUTLINE_INDICES

    # Use face mesh to get outline landmarks
    result_mesh = analyze_face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=face_outline_indices,
        image=image,
        face_mesh_obj=face_mesh_obj,
    )
    raw_landmarks = result_mesh.data

    if not raw_landmarks:
        return Result(image=None, data=None, meta={"source": image, "operation": "detect_faces", "max_faces": max_faces, "min_confidence": min_confidence, "error": "No face landmarks detected."})

    # Convert normalized coords to pixels
    all_polygons = []
    for face in raw_landmarks:
        polygon = [
            (int(x * width), int(y * height))
            for _, _, x, y, _ in face
        ]
        all_polygons.append(np.array(polygon, dtype=np.int32))

    # Crop faces
    cropped_faces = []
    for polygon in all_polygons:
        x, y, w, h = cv2.boundingRect(polygon)
        cropped = np_image[y:y + h, x:x + w]
        cropped_faces.append(cropped)

    return Result(image=cropped_faces, data=all_polygons, meta={"source": image, "operation": "detect_faces", "max_faces": max_faces, "min_confidence": min_confidence})
