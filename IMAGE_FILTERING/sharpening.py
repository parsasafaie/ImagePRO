import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import input/output managers and blur functions from your custom modules
from image_manager import input_manager, output_manger
from blur import average_blur


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

    Note:
        At least one of `image_path` or `np_image` must be provided.
    """
    # Load input image using input manager
    np_image = input_manager(image_path=image_path, np_image=np_image)

    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(np_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))  # Convert to valid pixel values

    # Enhance image by combining original with Laplacian response
    sharpen_image = np_image + laplacian_coefficient * laplacian
    sharpen_image = np.uint8(np.clip(sharpen_image, 0, 255))  # Clamp values to [0, 255]

    # Output the result (save or return)
    return output_manger(sharpen_image, result_path)


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

    Note:
        At least one of `image_path` or `np_image` must be provided.
    """
    # Load input image using input manager
    np_image = input_manager(image_path=image_path, np_image=np_image)

    # Blur the image using average blur function
    blur_image = average_blur(np_image=np_image)

    # Create mask by subtracting blurred image from original
    mask = cv2.subtract(np_image, blur_image)

    # Apply weighted addition to enhance sharpness
    sharpen_image = cv2.addWeighted(np_image, 1 + coefficient, mask, -coefficient, 0)

    # Output the result (save or return)
    return output_manger(sharpen_image, result_path)