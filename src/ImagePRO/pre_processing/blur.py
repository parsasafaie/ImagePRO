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


# Constants for blur operations
DEFAULT_KERNEL_SIZE = (5, 5)  # For average and Gaussian blur
DEFAULT_FILTER_SIZE = 5       # For median blur
DEFAULT_SIGMA_COLOR = 75      # For bilateral filter
DEFAULT_SIGMA_SPACE = 75      # For bilateral filter


def apply_average_blur(
    image: Image,
    *,
    kernel_size: tuple[int, int] = DEFAULT_KERNEL_SIZE
) -> Result:
    """
    Apply average (box) blur to an image.

    Uses a simple box filter where each output pixel is the mean of its kernel neighbors.
    Good for basic noise reduction but may create unwanted artifacts on edges.

    Args:
        image (Image):
            Input image to blur. Can be BGR (default) or RGB.
        kernel_size (tuple[int, int], optional):
            Blur kernel size as (width, height). Both must be positive integers.
            Larger values create stronger blur effect. Defaults to (5, 5).

    Returns:
        Result: Result object with blurred image.
            - image (np.ndarray): Blurred output image
            - data (None): No additional data
            - meta (dict): Contains kernel size and operation info

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If kernel_size contains invalid values
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance.")

    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or not all(isinstance(k, int) and k > 0 for k in kernel_size)
    ):
        raise ValueError("'kernel_size' must be a tuple of two positive integers.")

    # Apply box filter blur
    blurred = cv2.blur(image._data.copy(), kernel_size)
    
    return Result(
        image=blurred,
        meta={
            "source": image,
            "operation": "apply_average_blur",
            "kernel_size": kernel_size
        }
    )


def apply_gaussian_blur(
    image: Image,
    *,
    kernel_size: tuple[int, int] = DEFAULT_KERNEL_SIZE
) -> Result:
    """
    Apply Gaussian blur to an image.

    Uses a Gaussian kernel for smoother, more natural-looking blur than box filter.
    Ideal for general noise reduction and pre-processing for edge detection.

    Args:
        image (Image):
            Input image to blur. Can be BGR (default) or RGB.
        kernel_size (tuple[int, int], optional):
            Gaussian kernel size as (width, height). Both must be odd positive integers.
            Larger values create stronger blur effect. Defaults to (5, 5).

    Returns:
        Result: Result object with blurred image.
            - image (np.ndarray): Blurred output image
            - data (None): No additional data
            - meta (dict): Contains kernel size and operation info

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If kernel_size values are not odd positive integers
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance.")

    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or not all(isinstance(k, int) and k > 0 and k % 2 == 1 for k in kernel_size)
    ):
        raise ValueError("'kernel_size' must be a tuple of two odd positive integers.")

    # Apply Gaussian blur with automatic sigma calculation
    blurred = cv2.GaussianBlur(image._data.copy(), kernel_size, 0)
    
    return Result(
        image=blurred,
        meta={
            "source": image,
            "operation": "apply_gaussian_blur",
            "kernel_size": kernel_size
        }
    )


def apply_median_blur(
    image: Image,
    *,
    filter_size: int = DEFAULT_FILTER_SIZE
) -> Result:
    """
    Apply median blur to remove salt-and-pepper noise.

    Uses median filtering which is particularly effective at removing salt-and-pepper
    noise while preserving edges better than linear filters (average, Gaussian).

    Args:
        image (Image):
            Input image to blur. Can be BGR (default) or RGB.
        filter_size (int, optional):
            Size of the median filter kernel. Must be an odd integer > 1.
            Larger values create stronger noise reduction but slower processing.
            Defaults to 5.

    Returns:
        Result: Result object with denoised image.
            - image (np.ndarray): Denoised output image
            - data (None): No additional data
            - meta (dict): Contains filter size and operation info

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If filter_size is not an odd integer > 1
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance.")

    if not isinstance(filter_size, int) or filter_size <= 1 or filter_size % 2 == 0:
        raise ValueError("'filter_size' must be an odd integer greater than 1.")

    # Apply median filtering
    blurred = cv2.medianBlur(image._data.copy(), filter_size)
    
    return Result(
        image=blurred,
        meta={
            "source": image,
            "operation": "apply_median_blur",
            "filter_size": filter_size
        }
    )


def apply_bilateral_blur(
    image: Image,
    *,
    filter_size: int = 9,
    sigma_color: float = DEFAULT_SIGMA_COLOR,
    sigma_space: float = DEFAULT_SIGMA_SPACE
) -> Result:
    """
    Apply bilateral blur for edge-preserving smoothing.

    Uses bilateral filtering which smooths images while preserving edges by combining
    domain and range filtering. Effective for noise reduction while keeping sharp edges,
    but significantly slower than other blurring methods.

    Args:
        image (Image):
            Input image to blur. Can be BGR (default) or RGB.
        filter_size (int, optional):
            Diameter of pixel neighborhood. Larger values affect larger areas.
            Must be positive. Defaults to 9.
        sigma_color (float, optional):
            Filter sigma in color space. Larger values mean more dissimilar colors
            will be mixed together. Defaults to 75.
        sigma_space (float, optional):
            Filter sigma in coordinate space. Larger values mean more distant pixels
            will influence each other. Defaults to 75.

    Returns:
        Result: Result object with smoothed image.
            - image (np.ndarray): Edge-preserving smoothed output
            - data (None): No additional data
            - meta (dict): Contains filter *,parameters and operation info

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If any numeric parameter is not positive
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance.")

    if not isinstance(filter_size, int) or filter_size <= 0:
        raise ValueError("'filter_size' must be a positive integer.")
    if not isinstance(sigma_color, (int, float)) or sigma_color <= 0:
        raise ValueError("'sigma_color' must be a positive number.")
    if not isinstance(sigma_space, (int, float)) or sigma_space <= 0:
        raise ValueError("'sigma_space' must be a positive number.")

    # Apply bilateral filter
    blurred = cv2.bilateralFilter(
        image._data.copy(),
        filter_size,
        sigma_color,
        sigma_space
    )
    
    return Result(
        image=blurred,
        meta={
            "source": image,
            "operation": "apply_bilateral_blur",
            "filter_size": filter_size,
            "sigma_color": sigma_color,
            "sigma_space": sigma_space,
        }
    )
