from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

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
    color channel (BGR or RGB) with appropriate colors.

    The figure is closed before returning, so batch calls do not accumulate
    figures in memory. To display the plot interactively, call this function
    with ``matplotlib.interactive(True)`` or re-create the figure with
    ``plt.figure(1)`` right before ``plt.show()``: pyplot re-opens the most
    recently closed figure number with its contents intact.

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
        labels = ("Intensity",)
    else:
        raise ValueError(f"Unknown colorspace: {image.colorspace}")

    plt.figure(figsize=(10, 6))
    for channel, color in channels:
        hist = cv2.calcHist([image._data], [channel], None, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.title(f"Histogram of {image.colorspace} Channels")
    plt.legend(labels)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])

    result = Result(
        image=None,
        data=plt,
        meta={
            "source": image,
            "operation": "show_histogram"
        }
    )
    # Close the figure so repeated calls in batch jobs do not leak figures.
    plt.close()
    return result
