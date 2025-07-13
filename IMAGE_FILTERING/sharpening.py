import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler


def laplacian_filter(laplacian_coefficient=3, image_path=None, np_image=None, result_path=None):
    """
    Enhances image sharpness using the Laplacian filter method.
    
    Parameters:
        laplacian_coefficient (float): Strength of sharpening effect (higher = sharper).
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the sharpened image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the sharpened image as a NumPy array.

    Raises:
        ValueError: If values are out of valid range.
    """
    # Validate specific parameter
    if not isinstance(laplacian_coefficient, (int, float)) or laplacian_coefficient < 0:
        raise ValueError("'laplacian_coefficient' must be a non-negative number.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(np_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))  # Convert to valid pixel values

    # Enhance image by combining original with Laplacian response
    sharpen_image = np_image + laplacian_coefficient * laplacian
    sharpen_image = np.uint8(np.clip(sharpen_image, 0, 255))  # Clamp values to [0, 255]

    # Save or return
    return IOHandler.save_image(sharpen_image, result_path)


def unsharp_masking(coefficient=1, image_path=None, np_image=None, result_path=None):
    """
    Enhances image sharpness using Unsharp Masking technique.
    
    Parameters:
        coefficient (float): Strength of sharpening effect (higher = sharper).
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the sharpened image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the sharpened image as a NumPy array.

    Raises:
        ValueError: If values are out of valid range.
    """
    # Validate specific parameter
    if not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Blur the image using average blur function
    blur_image = average_blur(np_image=np_image)

    # Create mask by subtracting blurred image from original
    mask = cv2.subtract(np_image, blur_image)

    # Apply weighted addition to enhance sharpness
    sharpen_image = cv2.addWeighted(np_image, 1 + coefficient, mask, -coefficient, 0)

    # Save or return
    return IOHandler.save_image(sharpen_image, result_path)
