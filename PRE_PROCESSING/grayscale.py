import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler


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
    """
    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Convert image to grayscale using OpenCV's BGR to GRAY conversion
    grayscale_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    # Save or return
    return IOHandler.save_image(grayscale_image, result_path)
