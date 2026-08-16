from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


def show_histogram(image: Image) -> Result:
    """
    Displays the histogram of an image with customizable options.

    This function visualizes the intensity distribution of an image. It handles both grayscale
    and color images, with automatic detection of image type. For color images, it displays
    histograms for each color channel (BGR or RGB) with appropriate colors.

    Args:
        image (Image):
            Input image to convert. Must be BGR, RGB or Grayscale format.

    Returns:
        Result: Result object with histogram plot.
            - image (np.ndarray): None
            - data (None): matplotlib.pyplot object
            - meta (dict): Contains source object and operation info

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If image colorspace is not defined.
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance.")

    if image.colorspace in ("BGR", "RGB"):
        if image.colorspace == "BGR":
            colors = ("blue", "green", "red")
            labels = ("Blue Channel", "Green Channel", "Red Channel")
        else:
            colors = ("red", "green", "blue")
            labels = ("Red Channel", "Green Channel", "Blue Channel")
        channels = list(enumerate(colors))
    elif image.colorspace == "GRAY":
        channels = [(0, "black")]
        labels = None
    else:
        return ValueError("Unknown colorspace")

    plt.figure(figsize=(10, 6))
    for channel, color in channels:
        hist = cv2.calcHist([image._data], [channel], None, [256], [0, 256])
        plt.plot(hist, color=color)

    if labels is not None:
        plt.title(f"Histogram of {image.colorspace} Channels")
        plt.legend(labels)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

    plt.xlim([0, 256])

    return Result(
        image=None,
        data=plt,
        meta={
            "source": image,
            "operation": "show_histogram"
        }
    )
