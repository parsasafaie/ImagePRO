import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
DEFAULT_KERNEL_SIZE = (5, 5)
DEFAULT_FILTER_SIZE = 5
DEFAULT_SIGMA_COLOR = 75
DEFAULT_SIGMA_SPACE = 75


def apply_average_blur(
    *,
    image: Image | None = None,
    kernel_size: tuple[int, int] = DEFAULT_KERNEL_SIZE,
) -> np.ndarray:
    """
    Apply average blur (box filter) to an image.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to process.
    kernel_size : tuple[int, int], default=(5, 5)
        Blur kernel size (width, height). Both must be positive integers.

    Returns
    -------
    Result
        `image` is the blurred image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If kernel size or image is not valid.
    """
    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or not all(isinstance(k, int) and k > 0 for k in kernel_size)
    ):
        raise ValueError("'kernel_size' must be a tuple of two positive integers.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = image._data.copy()
    blurred = cv2.blur(annotated_image, kernel_size)
    
    return Result(image=blurred, data=None, meta={"source":image, "operation":"apply_average_blur", "kernel_size":kernel_size})


def apply_gaussian_blur(
    *,
    image: Image | None = None,
    kernel_size: tuple[int, int] = DEFAULT_KERNEL_SIZE,
) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to process.
    kernel_size : tuple[int, int], default=(5, 5)
        Kernel size (width, height), both odd positive integers.

    Returns
    -------
    Result
        `image` is the blurred image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If kernel size or image is invalid.
    """
    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or not all(isinstance(k, int) and k > 0 and k % 2 == 1 for k in kernel_size)
    ):
        raise ValueError("'kernel_size' must be a tuple of two odd positive integers.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = image._data.copy()
    blurred = cv2.GaussianBlur(annotated_image, kernel_size, 0)
    
    return Result(image=blurred, data=None, meta={"source":image, "operation":"apply_gaussian_blur", "kernel_size":kernel_size})


def apply_median_blur(
    *,
    image: Image | None = None,
    filter_size: int = DEFAULT_FILTER_SIZE,
) -> np.ndarray:
    """
    Apply median blur to remove salt-and-pepper noise.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to process.
    filter_size : int, default=5
        Must be an odd integer greater than 1.

    Returns
    -------
    Result
        `image` is the blurred image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If filter_size is not an odd integer greater than 1, or image is invalid.
    """
    if not isinstance(filter_size, int) or filter_size <= 1 or filter_size % 2 == 0:
        raise ValueError("'filter_size' must be an odd integer greater than 1.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = image._data.copy()
    blurred = cv2.medianBlur(annotated_image, filter_size)
    
    return Result(image=blurred, data=None, meta={"source":image, "operation":"apply_median_blur", "filter_size":filter_size})


def apply_bilateral_blur(
    *,
    image: Image | None = None,
    filter_size: int = 9,
    sigma_color: float = DEFAULT_SIGMA_COLOR,
    sigma_space: float = DEFAULT_SIGMA_SPACE,
) -> np.ndarray:
    """
    Apply bilateral blur to an image.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to process.
    filter_size : int, default=9
        Filter size, must be a positive integer.
    sigma_color : float, default=75
        Color sigma, higher values mean more colors will be mixed.
    sigma_space : float, default=75
        Space sigma, higher values mean farther pixels will influence each other.

    Returns
    -------
    Result
        `image` is the blurred image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If filter size, sigma parameters or image are invalid.
    """
    if not isinstance(filter_size, int) or filter_size <= 0:
        raise ValueError("'filter_size' must be a positive integer.")
    if not isinstance(sigma_color, (int, float)) or sigma_color <= 0:
        raise ValueError("'sigma_color' must be a positive number.")
    if not isinstance(sigma_space, (int, float)) or sigma_space <= 0:
        raise ValueError("'sigma_space' must be a positive number.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = image._data.copy()
    blurred = cv2.bilateralFilter(annotated_image, filter_size, sigma_color, sigma_space)
    
    return Result(image=blurred, data=None, meta={"source":image, "operation":"apply_bilateral_blur", "filter_size": filter_size, "sigma_code": sigma_color, "sigma_space": sigma_space})
