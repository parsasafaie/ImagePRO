import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import input/output managers from your custom module
from image_manager import input_manager, output_manger


def resize(new_size, image_path=None, np_image=None, result_path=None):
    """
    Resizes an image to the specified dimensions.
    
    Parameters:
        new_size (tuple): Target size as a tuple of (width, height).
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the resized image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the resized image as a NumPy array.

    Note:
        At least one of `image_path` or `np_image` must be provided.
    """
    # Load input image using input manager
    np_image = input_manager(image_path=image_path, np_image=np_image)

    # Resize the image to the specified dimensions
    resized_image = cv2.resize(np_image, dsize=new_size)

    # Output the result (save or return)
    return output_manger(resized_image, result_path)