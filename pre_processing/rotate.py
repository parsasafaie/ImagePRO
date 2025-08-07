import sys
from pathlib import Path

import cv2

# Add parent directory to sys.path for custom imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler


def rotate_image_90(image_path=None, np_image=None, result_path=None):
    """
    Rotate image 90 degrees clockwise.

    Args:
        image_path (str, optional): Path to input image file.
        np_image (np.ndarray, optional): Image as NumPy array.
        result_path (str, optional): Path to save rotated image.

    Returns:
        str or np.ndarray: Save message if result_path is given, else rotated image.
    """
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    rotated = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)
    return IOHandler.save_image(rotated, result_path)


def rotate_image_180(image_path=None, np_image=None, result_path=None):
    """
    Rotate image 180 degrees.

    Args:
        image_path (str, optional): Path to input image file.
        np_image (np.ndarray, optional): Image as NumPy array.
        result_path (str, optional): Path to save rotated image.

    Returns:
        str or np.ndarray: Save message if result_path is given, else rotated image.
    """
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    rotated = cv2.rotate(np_image, cv2.ROTATE_180)
    return IOHandler.save_image(rotated, result_path)


def rotate_image_270(image_path=None, np_image=None, result_path=None):
    """
    Rotate image 270 degrees clockwise (same as 90 degrees counter-clockwise).

    Args:
        image_path (str, optional): Path to input image file.
        np_image (np.ndarray, optional): Image as NumPy array.
        result_path (str, optional): Path to save rotated image.

    Returns:
        str or np.ndarray: Save message if result_path is given, else rotated image.
    """
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    rotated = cv2.rotate(np_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return IOHandler.save_image(rotated, result_path)


def rotate_image_custom(angle, scale=1.0, image_path=None, np_image=None, result_path=None):
    """
    Rotate image by custom angle around its center, with optional scaling.

    Args:
        angle (float): Rotation angle in degrees (positive = counter-clockwise).
        scale (float): Scale factor (default = 1.0).
        image_path (str, optional): Path to input image file.
        np_image (np.ndarray, optional): Image as NumPy array.
        result_path (str, optional): Path to save rotated image.

    Returns:
        str or np.ndarray: Save message if result_path is given, else rotated image.

    Raises:
        TypeError: If angle or scale are of incorrect type.
        ValueError: If scale is non-positive.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError("'angle' must be a number.")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("'scale' must be a positive number.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    height, width = np_image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated = cv2.warpAffine(np_image, matrix, (width, height))

    return IOHandler.save_image(rotated, result_path)
