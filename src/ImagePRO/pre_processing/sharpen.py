import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path for importing local modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

from ImagePRO.pre_processing.blur import apply_average_blur

# Constants
DEFAULT_LAPLACIAN_COEFFICIENT = 3.0
DEFAULT_UNSHARP_COEFFICIENT = 1.0


def apply_laplacian_sharpening(
    *,
    image: Image | None = None,
    coefficient: float = DEFAULT_LAPLACIAN_COEFFICIENT,
) -> np.ndarray:
    """
    Apply Laplacian filter to enhance image sharpness.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to convert.
    coefficient : float, default=3.0
        Intensity of sharpening effect (>= 0).

    Returns
    -------
    Result
        `image` is the sharpen image as a np.ndarray; `data` is None.
    
    Raises
    ------
    ValueError
        If coefficient is nan or negative or image is invalid.
    """
    if not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = image._data.copy()
    laplacian = cv2.Laplacian(annotated_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    sharpened = annotated_image + coefficient * laplacian
    sharpened = np.uint8(np.clip(sharpened, 0, 255))

    return Result(image=sharpened, data=None, meta={"source":image, "operation":"apply_laplacian_sharpening", "coefficient":coefficient})


def apply_unsharp_masking(
    *,
    image: Image | None = None,
    coefficient: float = DEFAULT_UNSHARP_COEFFICIENT,
) -> np.ndarray:
    """
    Apply Unsharp Masking to enhance image sharpness.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to convert.
    coefficient : float, default=3.0
        Intensity of sharpening effect (>= 0).

    Returns
    -------
    Result
        `image` is the sharpen image as a np.ndarray; `data` is None.
    
    Raises
    ------
    ValueError
        If coefficient is nan or negative or image is invalid.
    """
    if not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = image._data.copy()
    blurred = apply_average_blur(image=Image.from_array(annotated_image))

    mask = cv2.subtract(annotated_image, blurred)
    sharpened = cv2.addWeighted(annotated_image, 1 + coefficient, mask, -coefficient, 0)

    return Result(image=sharpened, data=None, meta={"source":image, "operation":"apply_unsharp_masking", "coefficient":coefficient})
