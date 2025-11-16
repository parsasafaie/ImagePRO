from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import cv2

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result
from ImagePRO.pre_processing.grayscale import convert_to_grayscale


# Constants for contrast enhancement
DEFAULT_CLIP_LIMIT = 2.0      # For CLAHE
DEFAULT_TILE_GRID_SIZE = (8, 8)  # For CLAHE
DEFAULT_ALPHA = 1.5           # For linear stretching
DEFAULT_BETA = 10            # For linear stretching


def apply_clahe_contrast(
    image: Image,
    *,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE
) -> Result:
    """Enhance image contrast using CLAHE (adaptive histogram equalization).

    CLAHE applies histogram equalization on small regions for better local contrast.
    Works well for images with varying lighting conditions.

    Args:
        image: Input image to enhance.
        clip_limit: Contrast threshold to prevent over-amplification. Must be > 0.
            Default: 2.0
        tile_grid_size: Size of grid for local histograms as (width, height).
            Default: (8, 8)

    Returns:
        Result object with enhanced image and metadata:
        - image: Contrast-enhanced image array
        - data: None
        - meta: Operation info and parameters used

    Raises:
        TypeError: If image is not an Image instance
        TypeError: If tile_grid_size is invalid
        ValueError: If clip_limit is not positive
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(clip_limit, (int, float)) or clip_limit <= 0:
        raise ValueError("'clip_limit' must be a positive number")

    if (
        not isinstance(tile_grid_size, tuple)
        or len(tile_grid_size) != 2
        or not all(isinstance(i, int) and i > 0 for i in tile_grid_size)
    ):
        raise TypeError("'tile_grid_size' must be a tuple of two positive integers")

    # Convert and apply CLAHE
    grayscale = convert_to_grayscale(image=Image.from_array(image._data.copy()))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(grayscale)

    return Result(
        image=enhanced,
        meta={
            "source": image,
            "operation": "apply_clahe_contrast",
            "clip_limit": clip_limit,
            "tile_grid_size": tile_grid_size
        }
    )


def apply_histogram_equalization(
    image: Image
) -> Result:
    """Global histogram equalization for contrast enhancement.

    Normalizes image intensity for better overall contrast.
    Simple but may overamplify noise in some cases.

    Args:
        image: Input image to enhance.

    Returns:
        Result object with enhanced image and metadata:
        - image: Contrast-enhanced image array
        - data: None
        - meta: Operation info

    Raises:
        TypeError: If image is not an Image instance
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    # Convert and apply global histogram equalization
    grayscale = convert_to_grayscale(image=Image.from_array(image._data.copy()))
    enhanced = cv2.equalizeHist(grayscale)

    return Result(
        image=enhanced,
        meta={
            "source": image,
            "operation": "apply_histogram_equalization"
        }
    )


def apply_contrast_stretching(
    image: Image,
    *,
    alpha: float = 1.0,
    beta: int = 130
) -> Result:
    """Linear contrast stretching using alpha and beta.

    Applies the formula: new_pixel = alpha × pixel + beta
    Simple but effective for basic contrast adjustment.

    Args:
        image: Input image to enhance.
        alpha: Multiplication factor (gain). Must be >= 0.
            Default: 1.0
        beta: Addition factor (bias). Must be 0-255.
            Default: 130

    Returns:
        Result object with enhanced image and metadata:
        - image: Contrast-enhanced image array
        - data: None
        - meta: Operation info and parameters used

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If alpha is negative
        ValueError: If beta is not in range 0-255
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if not isinstance(alpha, (int, float)) or alpha < 0:
        raise ValueError("'alpha' must be a non-negative number")

    if not isinstance(beta, int) or not (0 <= beta <= 255):
        raise ValueError("'beta' must be an integer between 0 and 255")

    # Convert and apply linear stretching
    grayscale = convert_to_grayscale(image=Image.from_array(image._data.copy()))
    enhanced = cv2.convertScaleAbs(grayscale, alpha=alpha, beta=beta)

    return Result(
        image=enhanced,
        meta={
            "source": image,
            "operation": "apply_contrast_stretching",
            "alpha": alpha,
            "beta": beta
        }
    )
