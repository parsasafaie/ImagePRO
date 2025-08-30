import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to sys.path for custom imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
DEFAULT_SCALE = 1.0
DEFAULT_ANGLE = 45.0


def rotate_image_90(
    *,
    image: Image | None = None,
) -> np.ndarray:
    """
    Rotate an image 90 degrees clockwise.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to rotate.

    Returns
    -------
    Result
        `image` is the rotated image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If image is invalid.
    """
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    
    annotated_image = image._data.copy()
    rotated = cv2.rotate(annotated_image, cv2.ROTATE_90_CLOCKWISE)
    return Result(image=rotated, data=None, meta={"source":image, "operation":"rotate_image_90"})


def rotate_image_180(
    *,
    image: Image | None = None,
) -> np.ndarray:
    """
    Rotate an image 180 degrees clockwise.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to rotate.

    Returns
    -------
    Result
        `image` is the rotated image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If image is invalid.
    """
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    
    annotated_image = image._data.copy()
    rotated = cv2.rotate(annotated_image, cv2.ROTATE_180)
    return Result(image=rotated, data=None, meta={"source":image, "operation":"rotate_image_180"})


def rotate_image_270(
    *,
    image: Image | None = None,
) -> np.ndarray:
    """
    Rotate an image 90 degrees clockwise.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to rotate.

    Returns
    -------
    Result
        `image` is the rotated image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If image is invalid.
    """
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    
    annotated_image = image._data.copy()
    rotated = cv2.rotate(annotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return Result(image=rotated, data=None, meta={"source":image, "operation":"rotate_image_270"})


def rotate_image_custom(
    *,
    image: Image | None = None,
    angle: float,
    scale: float = DEFAULT_SCALE,
) -> np.ndarray:
    """
    Rotate an image by a custom angle around its center with optional scaling.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to rotate.
    angle : float
        Rotation angle in degrees (positive = counter-clockwise).
    scale : float, default=1.0
        Scaling factor (> 0).

    Returns
    -------
    Result
        `image` is the rotated image as a np.ndarray; `data` is None.

    Raises
    ------
    TypeError
        If `angle` or `scale` are of incorrect type.
    ValueError
        If `scale` is not positive or image is invalid.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError("'angle' must be a number.")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("'scale' must be a positive number.")
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    annotated_image = image._data.copy()
    h, w = image.shape[:2]

    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated = cv2.warpAffine(annotated_image, matrix, (w, h))

    return Result(image=rotated, data=None, meta={"source":image, "operation":"rotate_image_custom", "angle":angle, "scale":scale})
