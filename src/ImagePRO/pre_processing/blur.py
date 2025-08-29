import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler

# Constants
DEFAULT_KERNEL_SIZE = (5, 5)
DEFAULT_FILTER_SIZE = 5
DEFAULT_SIGMA_COLOR = 75
DEFAULT_SIGMA_SPACE = 75


def apply_average_blur(
    kernel_size: tuple[int, int] = DEFAULT_KERNEL_SIZE,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Apply average blur (box filter) to an image.

    Parameters
    ----------
    kernel_size : tuple[int, int], default=(5, 5)
        Blur kernel size (width, height). Both must be positive integers.
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image array.
    output_image_path : str | None, optional
        If provided, save the blurred image.

    Returns
    -------
    np.ndarray
        Blurred image.

    Raises
    ------
    ValueError
        If kernel size is invalid.
    TypeError
        If input types are invalid.
    FileNotFoundError
        If image path does not exist.
    IOError
        If saving the image fails.
    """
    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or not all(isinstance(k, int) and k > 0 for k in kernel_size)
    ):
        raise ValueError("'kernel_size' must be a tuple of two positive integers.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    blurred = cv2.blur(np_image, kernel_size)
    
    if output_image_path:
        print(IOHandler.save_image(blurred, output_image_path))
    
    return blurred


def apply_gaussian_blur(
    kernel_size: tuple[int, int] = DEFAULT_KERNEL_SIZE,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Parameters
    ----------
    kernel_size : tuple[int, int], default=(5, 5)
        Kernel size (width, height), both odd positive integers.
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image array.
    output_image_path : str | None, optional
        If provided, save the blurred image.

    Returns
    -------
    np.ndarray
        Blurred image.

    Raises
    ------
    ValueError
        If kernel size is invalid.
    """
    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or not all(isinstance(k, int) and k > 0 and k % 2 == 1 for k in kernel_size)
    ):
        raise ValueError("'kernel_size' must be a tuple of two odd positive integers.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    blurred = cv2.GaussianBlur(np_image, kernel_size, 0)
    
    if output_image_path:
        print(IOHandler.save_image(blurred, output_image_path))
    
    return blurred


def apply_median_blur(
    filter_size: int = DEFAULT_FILTER_SIZE,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Apply median blur to remove salt-and-pepper noise.

    Parameters
    ----------
    filter_size : int, default=5
        Must be an odd integer greater than 1.
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image array.
    output_image_path : str | None, optional
        If provided, save the blurred image.

    Returns
    -------
    np.ndarray
        Blurred image.

    Raises
    ------
    TypeError
        If filter_size is not an integer.
    ValueError
        If filter_size is not an odd integer greater than 1.
    """
    if not isinstance(filter_size, int):
        raise TypeError("'filter_size' must be an integer.")
    if filter_size <= 1 or filter_size % 2 == 0:
        raise ValueError("'filter_size' must be an odd integer greater than 1.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    blurred = cv2.medianBlur(np_image, filter_size)
    
    if output_image_path:
        print(IOHandler.save_image(blurred, output_image_path))
    
    return blurred


def apply_bilateral_blur(
    filter_size: int = 9,
    sigma_color: float = DEFAULT_SIGMA_COLOR,
    sigma_space: float = DEFAULT_SIGMA_SPACE,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Apply bilateral blur to an image.

    Parameters
    ----------
    filter_size : int, default=9
        Filter size, must be a positive integer.
    sigma_color : float, default=75
        Color sigma, higher values mean more colors will be mixed.
    sigma_space : float, default=75
        Space sigma, higher values mean farther pixels will influence each other.
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image array.
    output_image_path : str | None, optional
        If provided, save the blurred image.

    Returns
    -------
    np.ndarray
        Blurred image.

    Raises
    ------
    ValueError
        If filter size or sigma parameters are invalid.
    """
    if not isinstance(filter_size, int) or filter_size <= 0:
        raise ValueError("'filter_size' must be a positive integer.")
    if not isinstance(sigma_color, (int, float)) or sigma_color <= 0:
        raise ValueError("'sigma_color' must be a positive number.")
    if not isinstance(sigma_space, (int, float)) or sigma_space <= 0:
        raise ValueError("'sigma_space' must be a positive number.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    blurred = cv2.bilateralFilter(np_image, filter_size, sigma_color, sigma_space)
    
    if output_image_path:
        print(IOHandler.save_image(blurred, output_image_path))
    
    return blurred
