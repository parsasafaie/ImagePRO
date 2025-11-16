from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
DEFAULT_START_POINT = (0, 0)
DEFAULT_END_POINT = (100, 100)


def crop_image(
    image: Image,
    *,
    start_point: tuple[int, int],
    end_point: tuple[int, int]
) -> Result:
    """Crop an image using top-left and bottom-right coordinates.

    Extracts a rectangular region from the image using the specified coordinates.
    The cropped area must be within image bounds and have valid dimensions.

    Args:
        image: Input image to crop.
        start_point: (x1, y1) coordinates of the top-left corner.
        end_point: (x2, y2) coordinates of the bottom-right corner.

    Returns:
        Result object with cropped image and metadata:
        - image: Cropped image array
        - meta: Operation info and coordinates used

    Raises:
        TypeError: If image is not an Image instance
        TypeError: If coordinates are not tuples of two integers
        ValueError: If coordinates are invalid or outside image bounds
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    # Validate coordinates format
    if (
        not isinstance(start_point, tuple) or
        not isinstance(end_point, tuple) or
        len(start_point) != 2 or len(end_point) != 2 or
        not all(isinstance(c, int) for c in start_point + end_point)
    ):
        raise TypeError("'start_point' and 'end_point' must be (x, y) tuples of integers")

    x1, y1 = start_point
    x2, y2 = end_point

    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop coordinates: ensure (x1, y1) is top-left and (x2, y2) is bottom-right")

    height, width = image.shape[:2]

    if x2 > width or y2 > height:
        raise ValueError(f"Crop area exceeds image bounds ({width}x{height})")

    # Extract the region
    cropped = image._data[y1:y2, x1:x2]

    return Result(
        image=cropped,
        meta={
            "source": image,
            "operation": "crop_image",
            "start_point": start_point,
            "end_point": end_point
        }
    )
