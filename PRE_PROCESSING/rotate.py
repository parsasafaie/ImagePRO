import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import input/output managers from your custom module
from image_manager import input_manager, output_manger


def rotate_90(image_path=None, np_image=None, result_path=None):
    """
    Rotates the image 90 degrees clockwise.
    
    Parameters:
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the rotated image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message. 
                          Otherwise, returns the rotated image as a NumPy array.

    Note:
        At least one of `image_path` or `np_image` must be provided.
    """
    np_image = input_manager(image_path=image_path, np_image=np_image)
    rotated_image = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)
    return output_manger(rotated_image, result_path)


def rotate_180(image_path=None, np_image=None, result_path=None):
    """
    Rotates the image 180 degrees.
    
    Parameters:
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the rotated image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message. 
                          Otherwise, returns the rotated image as a NumPy array.

    Note:
        At least one of `image_path` or `np_image` must be provided.
    """
    np_image = input_manager(image_path=image_path, np_image=np_image)
    rotated_image = cv2.rotate(np_image, cv2.ROTATE_180)
    return output_manger(rotated_image, result_path)


def rotate_270(image_path=None, np_image=None, result_path=None):
    """
    Rotates the image 270 degrees clockwise (same as 90 counter-clockwise).
    
    Parameters:
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the rotated image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message. 
                          Otherwise, returns the rotated image as a NumPy array.

    Note:
        At least one of `image_path` or `np_image` must be provided.
    """
    np_image = input_manager(image_path=image_path, np_image=np_image)
    rotated_image = cv2.rotate(np_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return output_manger(rotated_image, result_path)


def rotate_custom(angle, scale=1.0, image_path=None, np_image=None, result_path=None):
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

    Note:
        At least one of `image_path` or `np_image` must be provided.
    """
    np_image = input_manager(image_path=image_path, np_image=np_image)
    height, width = np_image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center=image_center, angle=angle, scale=scale)
    rotated_image = cv2.warpAffine(np_image, rotation_matrix, (width, height))
    return output_manger(rotated_image, result_path)