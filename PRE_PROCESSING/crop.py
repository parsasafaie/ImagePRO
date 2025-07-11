import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import input/output managers from your custom module
from image_manager import input_manager, output_manger


def crop(start_point, end_point, image_path=None, np_image=None, result_path=None):
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

    Note:
        At least one of `image_path` or `np_image` must be provided.
        Coordinates are zero-indexed and should be within image bounds.
    """
    # Load input image using input manager
    np_image = input_manager(image_path=image_path, np_image=np_image)

    # Unpack coordinates
    (x1, y1), (x2, y2) = start_point, end_point

    # Crop the image using NumPy slicing
    cropped_image = np_image[y1:y2, x1:x2]

    # Output the result (save or return)
    return output_manger(cropped_image, result_path)