import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to sys.path for custom imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler

# Constants
DEFAULT_SCALE = 1.0
DEFAULT_ANGLE = 45.0


def rotate_image_90(
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Rotate an image 90 degrees clockwise.

    Parameters
    ----------
    src_image_path : str | None, optional
        Path to the input image file.
    src_np_image : np.ndarray | None, optional
        Preloaded image array.
    output_image_path : str | None, optional
        Path to save the rotated image.

    Returns
    -------
    np.ndarray
        Rotated image.
    """
    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    rotated = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)
    if output_image_path:
        print(IOHandler.save_image(rotated, output_image_path))
    return rotated


def rotate_image_180(
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """Rotate an image 180 degrees."""
    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    rotated = cv2.rotate(np_image, cv2.ROTATE_180)
    if output_image_path:
        print(IOHandler.save_image(rotated, output_image_path))
    return rotated


def rotate_image_270(
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """Rotate an image 270 degrees clockwise (90° counter-clockwise)."""
    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    rotated = cv2.rotate(np_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if output_image_path:
        print(IOHandler.save_image(rotated, output_image_path))
    return rotated


def rotate_image_custom(
    angle: float,
    scale: float = DEFAULT_SCALE,
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Rotate an image by a custom angle around its center with optional scaling.

    Parameters
    ----------
    angle : float
        Rotation angle in degrees (positive = counter-clockwise).
    scale : float, default=1.0
        Scaling factor (> 0).
    src_image_path : str | None, optional
        Path to the input image file.
    src_np_image : np.ndarray | None, optional
        Preloaded image array.
    output_image_path : str | None, optional
        Path to save the rotated image.

    Returns
    -------
    np.ndarray
        Rotated image.

    Raises
    ------
    TypeError
        If `angle` or `scale` are of incorrect type.
    ValueError
        If `scale` is not positive.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError("'angle' must be a number.")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("'scale' must be a positive number.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    h, w = np_image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated = cv2.warpAffine(np_image, matrix, (w, h))
    if output_image_path:
        print(IOHandler.save_image(rotated, output_image_path))
    return rotated
