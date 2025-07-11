import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import input/output managers from your custom module
from image_manager import input_manager, output_manger


def grayscale(image_path=None, np_image=None, result_path=None):
    """
    Converts an image to grayscale (BGR to single-channel 8-bit image).
    
    Parameters:
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the grayscale image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the grayscale image as a NumPy array.

    Note:
        At least one of `image_path` or `np_image` must be provided.
    """
    # Load input image using input manager
    np_image = input_manager(image_path=image_path, np_image=np_image)

    # Convert image to grayscale using OpenCV's BGR to GRAY conversion
    grayscale_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    # Output the result (save or return)
    return output_manger(grayscale_image, result_path)