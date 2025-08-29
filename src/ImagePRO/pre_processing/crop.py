import sys
from pathlib import Path
import numpy as np

# Add parent directory to sys.path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler

# Constants
DEFAULT_START_POINT = (0, 0)
DEFAULT_END_POINT = (100, 100)


def crop_image(
    start_point: tuple[int, int],
    end_point: tuple[int, int],
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Crop an image using top-left and bottom-right coordinates.

    Parameters
    ----------
    start_point : tuple[int, int]
        (x1, y1) coordinates of the top-left corner.
    end_point : tuple[int, int]
        (x2, y2) coordinates of the bottom-right corner.
    src_image_path : str | None, optional
        Path to input image. Overrides `src_np_image` if provided.
    src_np_image : np.ndarray | None, optional
        Preloaded image array. Used if `src_image_path` is None.
    output_image_path : str | None, optional
        Path to save cropped image.

    Returns
    -------
    np.ndarray
        Cropped image.

    Raises
    ------
    TypeError
        If coordinates are not tuples of two integers.
    ValueError
        If coordinates are invalid or outside image bounds.
    FileNotFoundError
        If `src_image_path` does not exist.
    IOError
        If saving the image fails.
    """
    # Validate coordinates
    if (
        not isinstance(start_point, tuple) or
        not isinstance(end_point, tuple) or
        len(start_point) != 2 or len(end_point) != 2 or
        not all(isinstance(c, int) for c in start_point + end_point)
    ):
        raise TypeError("'start_point' and 'end_point' must be (x, y) tuples of integers.")

    x1, y1 = start_point
    x2, y2 = end_point

    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop coordinates: ensure (x1, y1) is top-left and (x2, y2) is bottom-right.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    height, width = np_image.shape[:2]

    if x2 > width or y2 > height:
        raise ValueError(f"Crop area exceeds image bounds ({width}x{height}).")

    cropped = np_image[y1:y2, x1:x2]

    if output_image_path:
        print(IOHandler.save_image(cropped, output_image_path))
    return cropped
