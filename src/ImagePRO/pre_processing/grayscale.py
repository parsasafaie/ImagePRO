import sys
from pathlib import Path

import cv2

# Add parent directory to sys.path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler


def convert_to_grayscale(image_path=None, np_image=None, result_path=None):
    """
    Convert a BGR image to single-channel grayscale.

    Args:
        image_path (str, optional): Path to input image file.
        np_image (np.ndarray, optional): Image array (used if image_path is None).
        result_path (str, optional): Path to save grayscale image.

    Returns:
        str or np.ndarray: Confirmation message if saved, else grayscale image.
    """
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    grayscale = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    return IOHandler.save_image(grayscale, result_path)
