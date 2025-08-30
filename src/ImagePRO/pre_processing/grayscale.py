from __future__ import annotations

import cv2

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


def convert_to_grayscale(image: Image) -> Result:
    """
    Convert an image to grayscale.

    Converts BGR or RGB image to single-channel grayscale using OpenCV.
    For BGR images (default), uses standard ITU-R BT.601 conversion.
    For RGB images, automatically handles colorspace conversion.

    Args:
        image (Image):
            Input image to convert. Must be BGR or RGB format.

    Returns:
        Result: Result object with converted grayscale image.
            - image (np.ndarray): Single-channel grayscale image
            - data (None): No additional data
            - meta (dict): Contains source object and operation info

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If image colorspace is already GRAY
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance.")

    if image.colorspace == "GRAY":
        raise ValueError("Image is already in grayscale format.")

    # Convert based on source colorspace
    if image.colorspace == "RGB":
        grayscale = cv2.cvtColor(image._data, cv2.COLOR_RGB2GRAY)
    else:  # BGR is default
        grayscale = cv2.cvtColor(image._data, cv2.COLOR_BGR2GRAY)

    return Result(
        image=grayscale,
        data=None,
        meta={
            "source": image,
            "operation": "convert_to_grayscale"
        }
    )
