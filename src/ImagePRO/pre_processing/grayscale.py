import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler


def convert_to_grayscale(
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Convert an image to grayscale.

    Parameters
    ----------
    src_image_path : str | None, optional
        Path to input image.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image array.
    output_image_path : str | None, optional
        If provided, save the grayscale image.

    Returns
    -------
    np.ndarray
        Grayscale image.

    Raises
    ------
    TypeError
        If input types are invalid.
    FileNotFoundError
        If image path does not exist.
    IOError
        If saving the image fails.
    """
    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    grayscale = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    
    if output_image_path:
        print(IOHandler.save_image(grayscale, output_image_path))
    
    return grayscale
