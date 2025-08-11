import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler


def apply_average_blur(kernel_size=(5, 5), image_path=None, np_image=None, result_path=None):
    """
    Apply average blur (box filter) to an image.

    Args:
        kernel_size (tuple): (width, height) of the blur kernel.
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Path to save output.

    Returns:
        str or np.ndarray: Saved message or blurred image.

    Raises:
        TypeError, ValueError: On invalid inputs.
    """
    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or not all(isinstance(k, int) and k > 0 for k in kernel_size)
    ):
        raise ValueError("'kernel_size' must be a tuple of two positive integers.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    blurred = cv2.blur(np_image, kernel_size)
    return IOHandler.save_image(blurred, result_path)


def apply_gaussian_blur(kernel_size=(5, 5), image_path=None, np_image=None, result_path=None):
    """
    Apply Gaussian blur to an image.

    Args:
        kernel_size (tuple): (width, height), both odd positive integers.
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Path to save output.

    Returns:
        str or np.ndarray: Saved message or blurred image.

    Raises:
        TypeError, ValueError: On invalid inputs.
    """
    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or not all(isinstance(k, int) and k > 0 and k % 2 == 1 for k in kernel_size)
    ):
        raise ValueError("'kernel_size' must be a tuple of two odd positive integers.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    blurred = cv2.GaussianBlur(np_image, kernel_size, sigmaX=0)
    return IOHandler.save_image(blurred, result_path)


def apply_median_blur(filter_size=5, image_path=None, np_image=None, result_path=None):
    """
    Apply median blur to remove salt-and-pepper noise.

    Args:
        filter_size (int): Must be an odd integer greater than 1.
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Path to save output.

    Returns:
        str or np.ndarray: Saved message or blurred image.

    Raises:
        TypeError, ValueError: On invalid inputs.
    """
    if not isinstance(filter_size, int):
        raise TypeError("'filter_size' must be an integer.")
    if filter_size <= 1 or filter_size % 2 == 0:
        raise ValueError("'filter_size' must be an odd integer greater than 1.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    blurred = cv2.medianBlur(np_image, filter_size)
    return IOHandler.save_image(blurred, result_path)


def apply_bilateral_blur(
    filter_size=9,
    sigma_color=75,
    sigma_space=75,
    image_path=None,
    np_image=None,
    result_path=None,
):
    """
    Apply bilateral filter to smooth while preserving edges.

    Args:
        filter_size (int): Diameter of pixel neighborhood.
        sigma_color (float): Color-space standard deviation.
        sigma_space (float): Coordinate-space standard deviation.
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Image array.
        result_path (str, optional): Path to save output.

    Returns:
        str or np.ndarray: Saved message or blurred image.

    Raises:
        TypeError, ValueError: On invalid inputs.
    """
    if not isinstance(filter_size, int) or filter_size < 1:
        raise ValueError("'filter_size' must be a positive integer.")
    if not isinstance(sigma_color, (int, float)) or sigma_color <= 0:
        raise ValueError("'sigma_color' must be a positive number.")
    if not isinstance(sigma_space, (int, float)) or sigma_space <= 0:
        raise ValueError("'sigma_space' must be a positive number.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    blurred = cv2.bilateralFilter(np_image, filter_size, sigma_color, sigma_space)
    return IOHandler.save_image(blurred, result_path)
