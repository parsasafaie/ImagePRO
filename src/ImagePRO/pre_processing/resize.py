import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to sys.path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler


def resize_image(
    new_size: tuple[int, int],
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Resize an image to the specified dimensions.

    Parameters
    ----------
    new_size : tuple[int, int]
        (width, height) in pixels, both must be positive integers.
    src_image_path : str | None, optional
        Path to the input image. Overrides `src_np_image` if provided.
    src_np_image : np.ndarray | None, optional
        Preloaded image array, used if `src_image_path` is None.
    output_image_path : str | None, optional
        Path to save the resized image.

    Returns
    -------
    np.ndarray
        Resized image.

    Raises
    ------
    TypeError
        If `new_size` is not a tuple of two integers.
    ValueError
        If width or height are not positive integers.
    FileNotFoundError
        If `src_image_path` does not exist.
    IOError
        If saving the image fails.
    """
    if (
        not isinstance(new_size, tuple)
        or len(new_size) != 2
        or not all(isinstance(x, int) for x in new_size)
    ):
        raise TypeError("'new_size' must be a tuple of two integers: (width, height).")

    if new_size[0] <= 0 or new_size[1] <= 0:
        raise ValueError("Both width and height must be positive integers.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    resized = cv2.resize(np_image, dsize=new_size)

    if output_image_path:
        print(IOHandler.save_image(resized, output_image_path))
    return resized
