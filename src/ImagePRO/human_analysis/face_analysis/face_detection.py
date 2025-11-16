from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[3]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import cv2
import mediapipe as mp
import numpy as np

from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh
from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


# Constants
DEFAULT_MAX_FACES = 1
DEFAULT_MIN_CONFIDENCE = 0.7
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 164, 163, 153, 157
]


def detect_faces(
    image: Image, 
    *,
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    face_mesh_obj: mp.solutions.face_mesh.FaceMesh | None = None
) -> Result:
    """Detect and crop face regions using facial landmarks.

    Uses face mesh to locate face outline points,
    then extracts rectangular regions containing each face.

    Args:
        image: Input image to process.
        max_faces: Maximum number of faces to detect.
            Must be positive.
            Default: 1
        min_confidence: Detection confidence threshold.
            Must be between 0 and 1.
            Default: 0.7
        face_mesh_obj: Pre-initialized face mesh detector.
            If None, creates new instance.
            Default: None

    Returns:
        Result object with detections:
        - image: List of cropped face images
            None if no faces detected
        - data: List of face outline polygons
            None if no faces detected
        - meta: Operation info and parameters
            Includes error info if detection fails

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If max_faces is not positive
        ValueError: If min_confidence not in [0,1]
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be positive")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1")

    # Get image dimensions
    height, width = image.shape[:2]

    # Get face landmarks
    result_mesh = analyze_face_mesh(
        image=image,
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=FACE_OUTLINE_INDICES,
        face_mesh_obj=face_mesh_obj
    )
    raw_landmarks = result_mesh.data

    # Handle no detections
    if not raw_landmarks:
        return Result(
            image=None,
            data=None,
            meta={
                "source": image,
                "operation": "detect_faces",
                "max_faces": max_faces,
                "min_confidence": min_confidence,
                "error": "No face landmarks detected"
            }
        )

    # Convert landmarks to pixel coordinates
    face_polygons = []
    for face in raw_landmarks:
        polygon = [
            (int(x * width), int(y * height))
            for _, _, x, y, _ in face
        ]
        face_polygons.append(np.array(polygon, dtype=np.int32))

    # Extract face regions
    face_regions = []
    for polygon in face_polygons:
        x, y, w, h = cv2.boundingRect(polygon)
        face_region = image._data[y:y + h, x:x + w]
        face_regions.append(face_region)

    return Result(
        image=face_regions,
        data=face_polygons,
        meta={
            "source": image,
            "operation": "detect_faces",
            "max_faces": max_faces,
            "min_confidence": min_confidence
        }
    )
