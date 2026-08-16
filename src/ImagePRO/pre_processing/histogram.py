from __future__ import annotations

import cv2
import matplotlib.pyplot as plt

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


def show_histogram(image: Image) -> Result:
    """
    Builds a matplotlib histogram figure for an image.

    This function creates the histogram plot of the intensity distribution of
    an image. It handles both grayscale and color images, with automatic
    detection of image type. For color images, it plots a histogram for each
    color channel (BGR or RGB) with appropriate colors. The pyplot module is
    returned in ``data``; call ``matplotlib.pyplot.show()`` to display the
    figure.

    Args:
        image (Image):
            Input image to analyze. Must be BGR, RGB or Grayscale format.

    Returns:
        Result: Result object with histogram plot.
            - image: None
            - data: The matplotlib.pyplot module (figure is created, not shown)
            - meta (dict): Contains source object and operation info

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If image colorspace is not supported.
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
