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
from ImagePRO.pre_processing.blur import apply_average_blur

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
        - image: Sharpened image array
        - data: None
        - meta: Operation info and coefficient used

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If coefficient is negative
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number")

    # Apply Laplacian edge detection
    laplacian = cv2.Laplacian(image._data.copy(), cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # Enhance edges
    sharpened = image._data + coefficient * laplacian
    sharpened = np.uint8(np.clip(sharpened, 0, 255))

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
        - image: Sharpened image array
        - data: None
        - meta: Operation info and coefficient used

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If coefficient is negative
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number")

    # Create the mask from original vs blurred difference
    blurred = apply_average_blur(image=Image.from_array(image._data.copy()))
    mask = cv2.subtract(image._data, blurred)
    
    # Apply unsharp masking
    sharpened = cv2.addWeighted(
        image._data, 
        1 + coefficient, 
        mask, 
        -coefficient, 
        0
    )

    return Result(
        image=sharpened,
        meta={
            "source": image,
            "operation": "apply_unsharp_masking",
            "coefficient": coefficient
        }
    )
