import cv2
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler


def resize_image(new_size, image_path=None, np_image=None, result_path=None):
    """
    Resizes an image to the specified dimensions.
    
    Parameters:
        new_size (tuple): Target size as a tuple of (width, height). Both must be positive integers.
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the resized image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the resized image as a NumPy array.

    Raises:
        TypeError: If `new_size` is not a tuple of two integers.
        ValueError: If width or height in `new_size` are not positive integers.
    """
    # Input validation specific to resize
    if not isinstance(new_size, tuple) or len(new_size) != 2:
        raise TypeError("'new_size' must be a tuple of two elements: (width, height).")

    if not isinstance(new_size[0], int) or not isinstance(new_size[1], int):
        raise TypeError("'width' and 'height' in 'new_size' must be integers.")

    if new_size[0] <= 0 or new_size[1] <= 0:
        raise ValueError("'width' and 'height' in 'new_size' must be positive integers.")

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Resize the image to the specified dimensions
    resized_image = cv2.resize(np_image, dsize=new_size)

    # Save or return
    return IOHandler.save_image(resized_image, result_path)
