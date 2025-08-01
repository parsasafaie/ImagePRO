import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler


def apply_average_blur(kernel_size=(5, 5), image_path=None, np_image=None, result_path=None):
    """
    Applies average blur to the image using a box filter.
    
    Parameters:
        kernel_size (tuple): Size of the kernel (width, height). Must be positive integers.
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the blurred image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the blurred image as a NumPy array.

    Raises:
        TypeError: If `kernel_size` is not valid.
        ValueError: If values are out of valid range.
    """
    # Validate specific parameters
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("'kernel_size' must be a tuple of two integers.")

    if not all(isinstance(x, int) and x > 0 for x in kernel_size):
        raise ValueError("'kernel_size' values must be positive integers.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Apply average blur
    blurred_image = cv2.blur(np_image, kernel_size)

    # Save or return
    return IOHandler.save_image(blurred_image, result_path)


def apply_gaussian_blur(kernel_size=(5, 5), image_path=None, np_image=None, result_path=None):
    """
    Applies Gaussian blur to the image.
    
    Parameters:
        kernel_size (tuple): Size of the Gaussian kernel (width, height). Must be positive odd integers.
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the blurred image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the blurred image as a NumPy array.

    Raises:
        TypeError: If `kernel_size` is not valid.
        ValueError: If values are out of valid range.
    """
    # Validate specific parameters
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("'kernel_size' must be a tuple of two integers.")

    if not all(isinstance(x, int) and x > 0 and x % 2 == 1 for x in kernel_size):
        raise ValueError("'kernel_size' values must be positive odd integers.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(np_image, kernel_size, 0)

    # Save or return
    return IOHandler.save_image(blurred_image, result_path)


def apply_median_blur(filter_size=5, image_path=None, np_image=None, result_path=None):
    """
    Applies median blur to reduce salt-and-pepper noise.
    
    Parameters:
        filter_size (int): Size of the filter kernel (must be odd and greater than 1).
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the blurred image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the blurred image as a NumPy array.

    Raises:
        TypeError: If `filter_size` is not an integer.
        ValueError: If `filter_size` is not valid.
    """
    # Validate specific parameter
    if not isinstance(filter_size, int):
        raise TypeError("'filter_size' must be an integer.")

    if filter_size <= 1 or filter_size % 2 == 0:
        raise ValueError("'filter_size' must be an odd integer greater than 1.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Apply median blur
    blurred_image = cv2.medianBlur(np_image, filter_size)

    # Save or return
    return IOHandler.save_image(blurred_image, result_path)


def apply_bilateral_blur(filter_size=9, sigma_color=75, sigma_space=75, image_path=None, np_image=None, result_path=None):
    """
    Applies bilateral filter to blur the image while preserving edges.
    
    Parameters:
        filter_size (int): Diameter of pixel neighborhood used for filtering.
        sigma_color (float): Filter sigma in the color space (larger = more smoothing).
        sigma_space (float): Filter sigma in the coordinate space (larger = more smoothing).
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the blurred image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the blurred image as a NumPy array.

    Raises:
        TypeError: If inputs are of incorrect type.
        ValueError: If values are out of valid range.
    """
    # Validate specific parameters
    if not isinstance(filter_size, int) or filter_size < 1:
        raise ValueError("'filter_size' must be a positive integer.")

    if not isinstance(sigma_color, (int, float)) or sigma_color <= 0:
        raise ValueError("'sigma_color' must be a positive number.")

    if not isinstance(sigma_space, (int, float)) or sigma_space <= 0:
        raise ValueError("'sigma_space' must be a positive number.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Apply bilateral filter
    blurred_image = cv2.bilateralFilter(np_image, d=filter_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Save or return
    return IOHandler.save_image(blurred_image, result_path)
