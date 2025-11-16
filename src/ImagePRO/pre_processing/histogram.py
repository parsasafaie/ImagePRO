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

    if image.colorspace == "BGR":
        colors = ('blue', 'green', 'red')
        plt.figure(figsize=(10, 6))

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image._data], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
            
        
        plt.title('Histogram of BGR Channels')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend(['Blue Channel', 'Green Channel', 'Red Channel'])
           
            
    elif image.colorspace == "RGB":
        colors = ('red', 'green', 'blue')
        plt.figure(figsize=(10, 6))

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image._data], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
            
        
        plt.title('Histogram of RGB Channels')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend(['Red Channel', 'Green Channel', 'Blue Channel'])
            
            
    elif image.colorspace == 'GRAY':
        plt.figure(figsize=(10, 6))

        hist = cv2.calcHist([image._data], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
        plt.xlim([0, 256])
        
    else:
        return ValueError('Unknown colorspace')
        

    return Result(
        image=None,
        data=plt,
        meta={
            "source": image,
            "operation": "show_histogram"
        }
    )
