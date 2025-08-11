import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to sys.path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler


def convert_to_grayscale(
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Convert a BGR image to single-channel grayscale.

    Parameters
    ----------
    src_image_path : str | None, optional
        Path to the input image file.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image. Used if `src_image_path` is None.
    output_image_path : str | None, optional
        Path to save the grayscale image.

    Returns
    -------
    np.ndarray
        Grayscale image.

    Raises
    ------
    TypeError
        If input types are invalid.
    FileNotFoundError
        If the provided `src_image_path` does not exist.
    IOError
        If saving the image fails.
    """
    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    grayscale = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    if output_image_path:
        print(IOHandler.save_image(grayscale, output_image_path))
    return grayscale
