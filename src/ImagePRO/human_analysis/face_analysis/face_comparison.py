from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[3]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import cv2
import numpy as np

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

if TYPE_CHECKING:  # insightface is imported lazily inside the function
    from insightface.app import FaceAnalysis

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
    and compares them using cosine similarity. Arrays are passed to the
    model in memory; only path-based images are read from disk, so no
    temporary files are written.

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
        FileNotFoundError: If a path-based image cannot be loaded
    """
    # Validate inputs
    if not isinstance(image_1, Image):
        raise TypeError("'image_1' must be an Image instance")
    if not isinstance(image_2, Image):
        raise TypeError("'image_2' must be an Image instance")

    try:
        from insightface.app import FaceAnalysis
    except ImportError as err:
        raise ImportError(
            "The optional 'insightface' dependency is required for face "
            'comparison. Install it with: pip install '
            '"ImagePRO-Python[insightface]"'
        ) from err

    # Initialize model if needed
    if app is None:
        app = FaceAnalysis(
            name=DEFAULT_MODEL_NAME,
            providers=[DEFAULT_PROVIDER]
        )
        app.prepare(ctx_id=0)  # Use CPU

    def load_rgb_image(image: Image) -> np.ndarray:
        """Return the image as an RGB array, via disk only if required."""
        if image.source_type == 'path':
            # honor the declared colorspace when reading from disk
            img = cv2.imread(str(image.path))
            if img is None:
                raise FileNotFoundError(f"Failed to load image from {image.path}")
            if image.colorspace == "RGB":
                return img
            if image.colorspace == "GRAY":
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # InsightFace accepts in-memory arrays; no temp file is written.
        if image.colorspace == "RGB":
            return image._data
        if image.colorspace == "GRAY":
            return cv2.cvtColor(image._data, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image._data, cv2.COLOR_BGR2RGB)

    img1 = load_rgb_image(image_1)
    img2 = load_rgb_image(image_2)

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
    # Cast to a plain bool: np.dot yields np.bool_, which fails `is True`
    # checks and serializes oddly in CSV/JSON output.
    is_match = bool(similarity > DEFAULT_SIMILARITY_THRESHOLD)

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
