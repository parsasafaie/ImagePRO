import sys
from pathlib import Path
import numpy as np

# Add parent directory to sys.path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
DEFAULT_START_POINT = (0, 0)
DEFAULT_END_POINT = (100, 100)


def crop_image(
    *,
    image: Image | None = None,
    start_point: tuple[int, int],
    end_point: tuple[int, int],
) -> np.ndarray:
    """
    Crop an image using top-left and bottom-right coordinates.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to crop.
    start_point : tuple[int, int]
        (x1, y1) coordinates of the top-left corner.
    end_point : tuple[int, int]
        (x2, y2) coordinates of the bottom-right corner.

    Returns
    -------
    Result
        `image` is the cropped image as a np.ndarray; `data` is None.

    Raises
    ------
    TypeError
        If coordinates are not tuples of two integers.
    ValueError
        If coordinates are invalid or outside image bounds, or image is invalid.
    """
    # Validate coordinates
    if (
        not isinstance(start_point, tuple) or
        not isinstance(end_point, tuple) or
        len(start_point) != 2 or len(end_point) != 2 or
        not all(isinstance(c, int) for c in start_point + end_point)
    ):
        raise TypeError("'start_point' and 'end_point' must be (x, y) tuples of integers.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    x1, y1 = start_point
    x2, y2 = end_point

    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop coordinates: ensure (x1, y1) is top-left and (x2, y2) is bottom-right.")

    annotated_image = image._data
    height, width = image.shape[:2]

    if x2 > width or y2 > height:
        raise ValueError(f"Crop area exceeds image bounds ({width}x{height}).")

    cropped = annotated_image[y1:y2, x1:x2]

    return Result(image=cropped, data=None, meta={"source":image, "operation":"crop_image", "start_point":start_point, "end_point":end_point})
