from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import cv2
import numpy as np

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


# Constants for rotation operations
DEFAULT_SCALE = 1.0
DEFAULT_ANGLE = 45.0


def rotate_image_90(image: Image) -> Result:
    """Rotate image 90 degrees clockwise.

    Args:
        image: Input image to rotate.

    Returns:
        Result object with rotated image and metadata:
        - image: Rotated image array
        - data: None
        - meta: Operation info and rotation angle

    Raises:
        TypeError: If image is not an Image instance
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    rotated = cv2.rotate(image._data.copy(), cv2.ROTATE_90_CLOCKWISE)
    return Result(
        image=rotated,
        meta={
            "source": image,
            "operation": "rotate_image_90",
            "angle": 90
        }
    )


def rotate_image_180(image: Image) -> Result:
    """Rotate image 180 degrees.

    Args:
        image: Input image to rotate.

    Returns:
        Result object with rotated image and metadata:
        - image: Rotated image array
        - data: None
        - meta: Operation info and rotation angle

    Raises:
        TypeError: If image is not an Image instance
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    rotated = cv2.rotate(image._data.copy(), cv2.ROTATE_180)
    return Result(
        image=rotated,
        meta={
            "source": image,
            "operation": "rotate_image_180",
            "angle": 180
        }
    )


def rotate_image_270(image: Image) -> Result:
    """Rotate image 270 degrees clockwise (90 counter-clockwise).

    Args:
        image: Input image to rotate.

    Returns:
        Result object with rotated image and metadata:
        - image: Rotated image array
        - data: None
        - meta: Operation info and rotation angle

    Raises:
        TypeError: If image is not an Image instance
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    rotated = cv2.rotate(image._data.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    return Result(
        image=rotated,
        meta={
            "source": image,
            "operation": "rotate_image_270",
            "angle": 270
        }
    )


def rotate_image_custom(
    image: Image,
    *,
    angle: float,
    scale: float = DEFAULT_SCALE
) -> Result:
    """Rotate image by custom angle with optional scaling.

    Rotates around image center. Positive angles are counter-clockwise.
    Image is resized to contain full rotated content.

    Args:
        image: Input image to rotate.
        angle: Rotation angle in degrees.
        scale: Image scaling factor, must be > 0. Default: 1.0

    Returns:
        Result object with rotated image and metadata:
        - image: Rotated and scaled image array
        - data: None
        - meta: Operation info, angle and scale values

    Raises:
        TypeError: If image is not an Image instance
        TypeError: If angle or scale are not numbers
        ValueError: If scale is not positive
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")
    if not isinstance(angle, (int, float)):
        raise TypeError("'angle' must be a number")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("'scale' must be a positive number")

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    rotated = cv2.warpAffine(image._data.copy(), matrix, (w, h))

    return Result(
        image=rotated,
        meta={
            "source": image,
            "operation": "rotate_image_custom",
            "angle": angle,
            "scale": scale
        }
    )
