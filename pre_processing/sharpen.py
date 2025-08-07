import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for importing local modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler
from blur import apply_average_blur


def apply_laplacian_sharpening(
    laplacian_coefficient=3, image_path=None, np_image=None, result_path=None
):
    """
    Apply Laplacian filter to enhance image sharpness.

    Args:
        laplacian_coefficient (float): Intensity of sharpening effect (>= 0).
        image_path (str, optional): Path to input image. Overrides `np_image` if given.
        np_image (np.ndarray, optional): Image as NumPy array.
        result_path (str, optional): Path to save the result.

    Returns:
        str or np.ndarray: Message if saved, else the sharpened image array.

    Raises:
        ValueError: If `laplacian_coefficient` is negative or inputs are invalid.
    """
    if not isinstance(laplacian_coefficient, (int, float)) or laplacian_coefficient < 0:
        raise ValueError("'laplacian_coefficient' must be a non-negative number.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    laplacian = cv2.Laplacian(np_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    sharpened = np_image + laplacian_coefficient * laplacian
    sharpened = np.uint8(np.clip(sharpened, 0, 255))

    return IOHandler.save_image(sharpened, result_path)


def apply_unsharp_masking(
    coefficient=1, image_path=None, np_image=None, result_path=None
):
    """
    Apply Unsharp Masking to enhance image sharpness.

    Args:
        coefficient (float): Intensity of sharpening effect (>= 0).
        image_path (str, optional): Path to input image. Overrides `np_image` if given.
        np_image (np.ndarray, optional): Image as NumPy array.
        result_path (str, optional): Path to save the result.

    Returns:
        str or np.ndarray: Message if saved, else the sharpened image array.

    Raises:
        ValueError: If `coefficient` is negative or inputs are invalid.
    """
    if not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    blurred = apply_average_blur(np_image=np_image)

    mask = cv2.subtract(np_image, blurred)
    sharpened = cv2.addWeighted(np_image, 1 + coefficient, mask, -coefficient, 0)

    return IOHandler.save_image(sharpened, result_path)
