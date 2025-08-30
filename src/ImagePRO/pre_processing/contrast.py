import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result
from ImagePRO.pre_processing.grayscale import convert_to_grayscale

# Constants
DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = (8, 8)
DEFAULT_ALPHA = 1.5
DEFAULT_BETA = 10


def apply_clahe_contrast(
    *,
    image: Image | None = None,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (adaptive histogram equalization).

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to process.
    clip_limit : float, default=2.0
        Contrast threshold (must be > 0).
    tile_grid_size : tuple[int, int], default=(8, 8)
        Grid size for local histogram (positive integers).

    Returns
    -------
    Result
        `image` is the enhanced image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If `clip_limit` <= 0 or image is invalid.
    TypeError
        If `tile_grid_size` is not valid.
    """
    if not isinstance(clip_limit, (int, float)) or clip_limit <= 0:
        raise ValueError("'clip_limit' must be a positive number.")

    if (
        not isinstance(tile_grid_size, tuple)
        or len(tile_grid_size) != 2
        or not all(isinstance(i, int) and i > 0 for i in tile_grid_size)
    ):
        raise TypeError("'tile_grid_size' must be a tuple of two positive integers.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = convert_to_grayscale(image=Image.from_array(image._data.copy()))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(annotated_image)

    return Result(image=enhanced, data=None, meta={"source":image, "operation":"apply_clahe_contrast", "clip_limit":clip_limit, "tile_grid_size":tile_grid_size})


def apply_histogram_equalization(
    *,
    image: Image | None = None,
) -> np.ndarray:
    """
    Enhance contrast using global histogram equalization.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to process.

    Returns
    -------
    Result
        `image` is the enhanced image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If image is invalid.
    """
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    
    annotated_image = convert_to_grayscale(image=Image.from_array(image._data.copy()))
    enhanced = cv2.equalizeHist(annotated_image)

    return Result(image=enhanced, data=None, meta={"source":image, "operation":"apply_histogram_equalization"})


def apply_contrast_stretching(
    *,
    image: Image | None = None,
    alpha: float = 1.0,
    beta: int = 130,
) -> np.ndarray:
    """
    Enhance contrast by linear stretching: `alpha × pixel + beta`.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to process.

    Returns
    -------
    Result
        `image` is the enhanced image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If `alpha` or `beta` are out of range, or image is invalid.
    """
    if not isinstance(alpha, (int, float)) or alpha < 0:
        raise ValueError("'alpha' must be a non-negative number.")

    if not isinstance(beta, int) or not (0 <= beta <= 255):
        raise ValueError("'beta' must be an integer between 0 and 255.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = convert_to_grayscale(image=Image.from_array(image._data.copy()))
    enhanced = cv2.convertScaleAbs(annotated_image, alpha=alpha, beta=beta)

    return Result(image=enhanced, data=None, meta={"source":image, "operation":"apply_contrast_stretching", "alpha":alpha, "beta":beta})
