import cv2
import numpy as np


def input_manager(image_path=None, np_image=None):
    """
    Manages image input by loading from a file path if provided,
    otherwise uses the given NumPy array image.

    Parameters:
        image_path (str): Path to the input image file (optional).
        np_image (np.ndarray): Pre-loaded image as a NumPy array (optional).

    Returns:
        np.ndarray: Loaded image as a NumPy array.
    
    Raises:
        ValueError: If both `image_path` and `np_image` are None.
        TypeError: If `image_path` is not a string or file not found.
        TypeError: If `np_image` is not a valid NumPy array.
    """
    # Input validation
    if image_path is None and np_image is None:
        raise ValueError("At least one of 'image_path' or 'np_image' must be provided.")

    if image_path is not None and not isinstance(image_path, str):
        raise TypeError("'image_path' must be a string representing the file path.")

    if np_image is not None and not isinstance(np_image, np.ndarray):
        raise TypeError("'np_image' must be a NumPy array.")

    # Load image
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found at '{image_path}'")
        return img
    else:
        return np_image


def output_manager(np_image, result_path=None):
    """
    Manages image output by either saving it to a file or returning it directly.

    Parameters:
        np_image (np.ndarray): Image to be outputted (as NumPy array).
        result_path (str): Path to save the image (optional).

    Returns:
        str | np.ndarray: File path confirmation message if saved, 
                          or the original image array if not saved.

    Raises:
        TypeError: If `np_image` is not a valid NumPy array.
        ValueError: If `result_path` is not a string when provided.
    """
    # Input validation
    if not isinstance(np_image, np.ndarray):
        raise TypeError("'np_image' must be a valid NumPy array.")

    if result_path is not None and not isinstance(result_path, str):
        raise ValueError("'result_path' must be a string representing the file path.")

    # Save or return image
    if result_path:
        success = cv2.imwrite(result_path, np_image)
        if not success:
            raise IOError(f"Failed to save image at '{result_path}'")
        return f"File saved at {result_path}"
    else:
        return np_image