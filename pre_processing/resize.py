import sys
from pathlib import Path

import cv2

# Add parent directory to sys.path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler


def resize_image(new_size, image_path=None, np_image=None, result_path=None):
    """
    Resize image to specified dimensions.

    Args:
        new_size (tuple): (width, height), both positive integers.
        image_path (str, optional): Path to input image. Overrides `np_image` if given.
        np_image (np.ndarray, optional): Image array (used if image_path is None).
        result_path (str, optional): Save path. If None, returns resized image.

    Returns:
        str or np.ndarray: Confirmation message if saved, else resized image.

    Raises:
        TypeError: If new_size is not a tuple of two integers.
        ValueError: If width or height are not positive.
    """
    if (
        not isinstance(new_size, tuple) or
        len(new_size) != 2 or
        not all(isinstance(x, int) for x in new_size)
    ):
        raise TypeError("'new_size' must be a tuple of two integers: (width, height).")

    if new_size[0] <= 0 or new_size[1] <= 0:
        raise ValueError("Both width and height must be positive integers.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    resized = cv2.resize(np_image, dsize=new_size)

    return IOHandler.save_image(resized, result_path)
