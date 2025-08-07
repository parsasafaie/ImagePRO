import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from pre_processing.grayscale import convert_to_grayscale
from io_handler import IOHandler


def apply_clahe_contrast(
    clipLimit=2.0,
    tileGridSize=(8, 8),
    image_path=None,
    np_image=None,
    result_path=None,
):
    """
    Enhance contrast using CLAHE (adaptive histogram equalization).

    Args:
        clipLimit (float): Contrast threshold (must be > 0).
        tileGridSize (tuple): Grid size for local histogram (e.g., (8, 8)).
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Path to save result.

    Returns:
        str or np.ndarray: Confirmation if saved, else enhanced image.

    Raises:
        ValueError, TypeError: On invalid parameter values.
    """
    if not isinstance(clipLimit, (int, float)) or clipLimit <= 0:
        raise ValueError("'clipLimit' must be a positive number.")

    if (
        not isinstance(tileGridSize, tuple)
        or len(tileGridSize) != 2
        or not all(isinstance(i, int) and i > 0 for i in tileGridSize)
    ):
        raise TypeError("'tileGridSize' must be a tuple of two positive integers.")

    np_image = convert_to_grayscale(np_image=IOHandler.load_image(image_path=image_path, np_image=np_image))

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    enhanced = clahe.apply(np_image)

    return IOHandler.save_image(enhanced, result_path)


def apply_histogram_equalization(image_path=None, np_image=None, result_path=None):
    """
    Enhance contrast using global histogram equalization.

    Args:
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Path to save result.

    Returns:
        str or np.ndarray: Confirmation if saved, else enhanced image.
    """
    np_image = convert_to_grayscale(np_image=IOHandler.load_image(image_path=image_path, np_image=np_image))
    enhanced = cv2.equalizeHist(np_image)
    return IOHandler.save_image(enhanced, result_path)


def apply_contrast_stretching(
    alpha,
    beta,
    image_path=None,
    np_image=None,
    result_path=None,
):
    """
    Enhance contrast by linear contrast stretching (alpha × pixel + beta).

    Args:
        alpha (float): Contrast factor (>= 0).
        beta (int): Brightness offset (0–255).
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Path to save result.

    Returns:
        str or np.ndarray: Confirmation if saved, else enhanced image.

    Raises:
        ValueError: If parameters are out of range.
    """
    if not isinstance(alpha, (int, float)) or alpha < 0:
        raise ValueError("'alpha' must be a non-negative number.")

    if not isinstance(beta, int) or not (0 <= beta <= 255):
        raise ValueError("'beta' must be an integer between 0 and 255.")

    np_image = convert_to_grayscale(np_image=IOHandler.load_image(image_path=image_path, np_image=np_image))
    enhanced = cv2.convertScaleAbs(np_image, alpha=alpha, beta=beta)

    return IOHandler.save_image(enhanced, result_path)
