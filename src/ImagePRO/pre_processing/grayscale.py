import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


def convert_to_grayscale(
    *,
    image: Image | None = None,
) -> np.ndarray:
    """
    Convert an image to grayscale.

    Parameters
    ----------
    image : Image
        Image instance (BGR data expected) to convert.

    Returns
    -------
    Result
        `image` is the converted image as a np.ndarray; `data` is None.

    Raises
    ------
    ValueError
        If image is invalid.
    """
    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    
    annotated_image = image._date
    grayscale = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
    
    return Result(image=grayscale, data=None, meta={"source":image, "operation":"convert_to_grayscale"})
