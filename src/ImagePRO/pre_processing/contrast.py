import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from pre_processing.grayscale import convert_to_grayscale
from utils.io_handler import IOHandler

# Constants
DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = (8, 8)
DEFAULT_ALPHA = 1.5
DEFAULT_BETA = 10


def apply_clahe_contrast(
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (adaptive histogram equalization).

    Parameters
    ----------
    clip_limit : float, default=2.0
        Contrast threshold (must be > 0).
    tile_grid_size : tuple[int, int], default=(8, 8)
        Grid size for local histogram (positive integers).
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded image.
    output_image_path : str | None, optional
        Path to save the enhanced image.

    Returns
    -------
    np.ndarray
        Enhanced grayscale image.

    Raises
    ------
    ValueError
        If `clip_limit` <= 0.
    TypeError
        If `tile_grid_size` is not valid.
    FileNotFoundError
        If image path is invalid.
    IOError
        If saving the image fails.
    """
    if not isinstance(clip_limit, (int, float)) or clip_limit <= 0:
        raise ValueError("'clip_limit' must be a positive number.")

    if (
        not isinstance(tile_grid_size, tuple)
        or len(tile_grid_size) != 2
        or not all(isinstance(i, int) and i > 0 for i in tile_grid_size)
    ):
        raise TypeError("'tile_grid_size' must be a tuple of two positive integers.")

    np_image = convert_to_grayscale(
        np_image=IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    )

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(np_image)

    if output_image_path:
        print(IOHandler.save_image(enhanced, output_image_path))
    return enhanced


def apply_histogram_equalization(
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Enhance contrast using global histogram equalization.

    Parameters
    ----------
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded image.
    output_image_path : str | None, optional
        Path to save the enhanced image.

    Returns
    -------
    np.ndarray
        Enhanced grayscale image.
    """
    np_image = convert_to_grayscale(
        np_image=IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    )
    enhanced = cv2.equalizeHist(np_image)

    if output_image_path:
        print(IOHandler.save_image(enhanced, output_image_path))
    return enhanced


def apply_contrast_stretching(
    alpha: float,
    beta: int,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Enhance contrast by linear stretching: `alpha × pixel + beta`.

    Parameters
    ----------
    alpha : float
        Contrast factor (>= 0).
    beta : int
        Brightness offset (0–255).
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded image.
    output_image_path : str | None, optional
        Path to save the enhanced image.

    Returns
    -------
    np.ndarray
        Enhanced grayscale image.

    Raises
    ------
    ValueError
        If `alpha` or `beta` are out of range.
    """
    if not isinstance(alpha, (int, float)) or alpha < 0:
        raise ValueError("'alpha' must be a non-negative number.")

    if not isinstance(beta, int) or not (0 <= beta <= 255):
        raise ValueError("'beta' must be an integer between 0 and 255.")

    np_image = convert_to_grayscale(
        np_image=IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    )
    enhanced = cv2.convertScaleAbs(np_image, alpha=alpha, beta=beta)

    if output_image_path:
        print(IOHandler.save_image(enhanced, output_image_path))
    return enhanced
