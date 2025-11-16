from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import cv2

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


def resize_image(
    image: Image,
    *,
    new_size: tuple[int, int]
) -> Result:
    """Resize an image to specified dimensions.

    Changes image size while maintaining aspect ratio.
    Uses bilinear interpolation for smooth resizing.

    Args:
        image: Input image to resize.
        new_size: Target size as (width, height) in pixels.
            Both dimensions must be positive integers.

    Returns:
        Result object with resized image and metadata:
        - image: Resized image array
        - data: None
        - meta: Operation info and new size used

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If new_size is not valid (tuple of 2 positive ints)
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    if (
        not isinstance(new_size, tuple)
        or len(new_size) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in new_size)
    ):
        raise ValueError("'new_size' must be a tuple of two positive integers")

    # Resize using bilinear interpolation
    resized = cv2.resize(image._data, new_size, interpolation=cv2.INTER_LINEAR)
    
    return Result(
        image=resized,
        meta={
            "source": image,
            "operation": "resize_image",
            "new_size": new_size
        }
    )
