import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


def resize_image(
    *,
    image: Image | None = None,
    new_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize an image to new dimensions.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to convert.
    new_size : tuple[int, int]
        New image size as (width, height).

    Returns
    -------
    Result
        `image` is the resized image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If new_size or image are invalid.
    TypeError
        If input types are invalid.
    """
    if (
        not isinstance(new_size, tuple)
        or len(new_size) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in new_size)
    ):
        raise ValueError("'new_size' must be a tuple of two positive integers.")
    
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = image._data
    resized = cv2.resize(annotated_image, new_size, interpolation=cv2.INTER_LINEAR)
    
    return Result(image=resized, data=None, meta={"source":image, "operation":"resize_image", "new_size":new_size})
