import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler


def crop_image(start_point, end_point, image_path=None, np_image=None, result_path=None):
    """
    Crops an image between two given points.
    
    Parameters:
        start_point (tuple): Starting point as (x, y) - top-left corner.
        end_point (tuple): Ending point as (x, y) - bottom-right corner.
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the cropped image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the cropped image as a NumPy array.

    Raises:
        TypeError: If `start_point` or `end_point` are not tuples of two integers.
        ValueError: If coordinates are invalid (negative values or out of bounds).
    """
    # Validate specific parameters
    if not isinstance(start_point, tuple) or len(start_point) != 2:
        raise TypeError("'start_point' must be a tuple of two integers (x, y).")

    if not isinstance(end_point, tuple) or len(end_point) != 2:
        raise TypeError("'end_point' must be a tuple of two integers (x, y).")

    if not all(isinstance(coord, int) for coord in start_point + end_point):
        raise TypeError("All coordinates in 'start_point' and 'end_point' must be integers.")

    x1, y1 = start_point
    x2, y2 = end_point

    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValueError("All coordinates must be non-negative.")

    if x1 >= x2 or y1 >= y2:
        raise ValueError("'start_point' must be top-left and 'end_point' bottom-right. "
                         "Ensure x1 < x2 and y1 < y2.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Get image dimensions
    height, width = np_image.shape[:2]

    # Check if coordinates are within image bounds
    if x2 > width or y2 > height:
        raise ValueError(f"Crop area ({x1}, {y1}) to ({x2}, {y2}) exceeds image size ({width}x{height}).")

    # Crop the image using NumPy slicing
    cropped_image = np_image[y1:y2, x1:x2]

    # Save or return
    return IOHandler.save_image(cropped_image, result_path)
