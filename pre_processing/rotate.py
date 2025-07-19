import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler


def rotate_image_90(image_path=None, np_image=None, result_path=None):
    """
    Rotates the image 90 degrees clockwise.
    
    Parameters:
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the rotated image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message. 
                          Otherwise, returns the rotated image as a NumPy array.
    """
    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Rotate
    rotated_image = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)

    # Save or return
    return IOHandler.save_image(rotated_image, result_path)


def rotate_image_180(image_path=None, np_image=None, result_path=None):
    """
    Rotates the image 180 degrees.
    
    Parameters:
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the rotated image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message. 
                          Otherwise, returns the rotated image as a NumPy array.
    """
    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Rotate
    rotated_image = cv2.rotate(np_image, cv2.ROTATE_180)

    # Save or return
    return IOHandler.save_image(rotated_image, result_path)


def rotate_image_270(image_path=None, np_image=None, result_path=None):
    """
    Rotates the image 270 degrees clockwise (same as 90 counter-clockwise).
    
    Parameters:
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the rotated image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message. 
                          Otherwise, returns the rotated image as a NumPy array.
    """
    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Rotate
    rotated_image = cv2.rotate(np_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Save or return
    return IOHandler.save_image(rotated_image, result_path)


def rotate_image_custom(angle, scale=1.0, image_path=None, np_image=None, result_path=None):
    """
    Rotates image by a given angle around its center with optional scaling.
    
    Parameters:
        angle (float): Rotation angle in degrees (positive is counter-clockwise).
        scale (float): Optional scale factor (default = 1.0).
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the rotated image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message. 
                          Otherwise, returns the rotated image as a NumPy array.

    Raises:
        TypeError: If inputs are of incorrect type.
        ValueError: If inputs have invalid values.
    """
    # Input validation - only specific parameters
    if not isinstance(angle, (int, float)):
        raise TypeError("'angle' must be a number (int or float).")

    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("'scale' must be a positive number.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Rotate
    height, width = np_image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center=image_center, angle=angle, scale=scale)
    rotated_image = cv2.warpAffine(np_image, rotation_matrix, (width, height))

    # Save or return
    return IOHandler.save_image(rotated_image, result_path)
