from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[3]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_PROVIDER = "CPUExecutionProvider"


def compare_faces(
    image_1: Image,
    image_2: Image,
    *,
    app: FaceAnalysis | None = None
) -> Result:
    """Compare two face images to determine if they are the same person.

    Uses InsightFace's FaceAnalysis model to extract facial embeddings
    and compares them using cosine similarity.

    Args:
        image_1: First image to compare.
            Must contain a clearly visible face.
        image_2: Second image to compare.
            Must contain a clearly visible face.
        app: Pre-initialized FaceAnalysis model.
            If None, creates new instance.
            Default: None

    Returns:
        Result object with comparison results:
        - image: None
        - data: True if same person, False if different
            None if face detection fails
        - meta: Operation info and similarity score
            Error details if detection fails

    Raises:
        TypeError: If either image is not an Image instance
        FileNotFoundError: If image data cannot be loaded
    """
    # Validate inputs
    if not isinstance(image_1, Image):
        raise TypeError("'image_1' must be an Image instance")
    if not isinstance(image_2, Image):
        raise TypeError("'image_2' must be an Image instance")

    # Initialize model if needed
    if app is None:
        app = FaceAnalysis(
            name=DEFAULT_MODEL_NAME,
            providers=[DEFAULT_PROVIDER]
        )
        app.prepare(ctx_id=0)  # Use CPU

    def save_temp_image(image: Image, temp_name: str) -> str:
        """Save image data to temporary file if needed."""
        if image.source_type == 'path':
            return image.path
        temp_path = os.path.join(os.getcwd(), temp_name)
        cv2.imwrite(temp_path, image._data)
        return temp_path

    def load_rgb_image(path: str) -> np.ndarray | None:
        """Load image file and convert to RGB."""
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    try:
        # Save images to disk if needed
        path_1 = save_temp_image(image_1._data, 'tmp1.jpg')
        path_2 = save_temp_image(image_2._data, 'tmp2.jpg')

        # Load and preprocess images
        img1 = load_rgb_image(path_1)
        img2 = load_rgb_image(path_2)

        if img1 is None or img2 is None:
            raise FileNotFoundError("Failed to load one or both images")

        # Detect faces
        faces1 = app.get(img1)
        faces2 = app.get(img2)

        # Validate detections
        if not faces1 or not faces2:
            return Result(
                image=None,
                data=None,
                meta={
                    "source": (image_1, image_2),
                    "operation": "compare_faces",
                    "error": "No face detected in one or both images"
                }
            )

        # Extract embeddings
        emb1 = faces1[0].embedding
        emb2 = faces2[0].embedding

        # Calculate similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        is_match = similarity > DEFAULT_SIMILARITY_THRESHOLD

        return Result(
            image=None,
            data=is_match,
            meta={
                "source": (image_1, image_2),
                "operation": "compare_faces",
                "similarity": float(similarity),
                "threshold": DEFAULT_SIMILARITY_THRESHOLD
            }
        )

    finally:
        # Cleanup temporary files
        if image_1.source_type != 'path' and os.path.exists('tmp1.jpg'):
            os.remove('tmp1.jpg')
        if image_2.source_type != 'path' and os.path.exists('tmp2.jpg'):
            os.remove('tmp2.jpg')
