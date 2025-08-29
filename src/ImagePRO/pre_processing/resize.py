import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler


def resize_image(
    new_size: tuple[int, int],
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Resize an image to new dimensions.

    Parameters
    ----------
    new_size : tuple[int, int]
        New image size as (width, height).
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image array.
    output_image_path : str | None, optional
        If provided, save the resized image.

    Returns
    -------
    np.ndarray
        Resized image.

    Raises
    ------
    ValueError
        If new_size is invalid.
    TypeError
        If input types are invalid.
    FileNotFoundError
        If image path does not exist.
    IOError
        If saving the image fails.
    """
    if (
        not isinstance(new_size, tuple)
        or len(new_size) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in new_size)
    ):
        raise ValueError("'new_size' must be a tuple of two positive integers.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    resized = cv2.resize(np_image, new_size, interpolation=cv2.INTER_LINEAR)
    
    if output_image_path:
        print(IOHandler.save_image(resized, output_image_path))
    
    return resized
