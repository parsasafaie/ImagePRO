from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import cv2
import numpy as np

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result
from ImagePRO.pre_processing.blur import DEFAULT_KERNEL_SIZE

# Constants
DEFAULT_LAPLACIAN_COEFFICIENT = 3.0
DEFAULT_UNSHARP_COEFFICIENT = 1.0


def apply_laplacian_sharpening(
    image: Image,
    *,
    coefficient: float = DEFAULT_LAPLACIAN_COEFFICIENT
) -> Result:
    """Enhance image sharpness using Laplacian filtering.

    Applies edge detection and enhances edges to improve sharpness.
    Useful for bringing out fine details in images.

    Args:
        image: Input image to sharpen.
        coefficient: Intensity of sharpening effect. Must be >= 0.
            Default: 3.0

    Returns:
        Result object with sharpened image and metadata:
        - image: Sharpened image array (same dtype as the input)
        - data: None
        - meta: Operation info and coefficient used

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If coefficient is negative
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) \
            or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number")

    # Sharpen in float space: image + coefficient * |laplacian|, clipped
    # once at the end. Casting the Laplacian to uint8 before combining
    # would truncate edge responses above 255 and corrupt the output.
    laplacian = cv2.Laplacian(image._data, cv2.CV_64F)
    np.absolute(laplacian, out=laplacian)

    sharpened = image._data.astype(np.float64) + coefficient * laplacian
    np.clip(sharpened, 0, 255, out=sharpened)
    sharpened = sharpened.astype(image._data.dtype)

    return Result(
        image=sharpened,
        meta={
            "source": image,
            "operation": "apply_laplacian_sharpening",
            "coefficient": coefficient
        }
    )


def apply_unsharp_masking(
    image: Image,
    *,
    coefficient: float = DEFAULT_UNSHARP_COEFFICIENT
) -> Result:
    """Enhance image sharpness using unsharp masking.

    Creates a blurred version, subtracts from original to get edges,
    then enhances those edges in the original image.

    Args:
        image: Input image to sharpen.
        coefficient: Intensity of sharpening effect. Must be >= 0.
            Default: 1.0

    Returns:
        Result object with sharpened image and metadata:
        - image: Sharpened image array (same dtype as the input)
        - data: None
        - meta: Operation info and coefficient used

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If coefficient is negative
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) \
            or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number")

    # Unsharp masking: original + coefficient * (original - blurred),
    # computed in float space and clipped once at the end. cv2.subtract
    # saturates negative mask values and addWeighted on the saturated mask
    # brightens flat regions instead of sharpening, so both are avoided.
    blurred = cv2.blur(image._data, DEFAULT_KERNEL_SIZE)
    mask = image._data.astype(np.float64) - blurred.astype(np.float64)

    sharpened = image._data.astype(np.float64) + coefficient * mask
    np.clip(sharpened, 0, 255, out=sharpened)
    sharpened = sharpened.astype(image._data.dtype)

    return Result(
        image=sharpened,
        meta={
            "source": image,
            "operation": "apply_unsharp_masking",
            "coefficient": coefficient
        }
    )
