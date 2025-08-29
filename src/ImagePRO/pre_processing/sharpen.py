import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path for importing local modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler
from pre_processing.blur import apply_average_blur

# Constants
DEFAULT_LAPLACIAN_COEFFICIENT = 3.0
DEFAULT_UNSHARP_COEFFICIENT = 1.0


def apply_laplacian_sharpening(
    coefficient: float = DEFAULT_LAPLACIAN_COEFFICIENT,
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Apply Laplacian filter to enhance image sharpness.

    Parameters
    ----------
    coefficient : float, default=3.0
        Intensity of sharpening effect (>= 0).
    src_image_path : str | None
        Path to the input image file.
    src_np_image : np.ndarray | None
        Preloaded image array.
    output_image_path : str | None
        Path to save the sharpened image.

    Returns
    -------
    np.ndarray
        Sharpened image.
    """
    if not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    laplacian = cv2.Laplacian(np_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    sharpened = np_image + coefficient * laplacian
    sharpened = np.uint8(np.clip(sharpened, 0, 255))

    if output_image_path:
        print(IOHandler.save_image(sharpened, output_image_path))
    return sharpened


def apply_unsharp_masking(
    coefficient: float = DEFAULT_UNSHARP_COEFFICIENT,
    src_image_path: str | None = None,
    src_np_image: np.ndarray | None = None,
    output_image_path: str | None = None
) -> np.ndarray:
    """
    Apply Unsharp Masking to enhance image sharpness.

    Parameters
    ----------
    coefficient : float, default=1.0
        Intensity of sharpening effect (>= 0).
    src_image_path : str | None
        Path to the input image file.
    src_np_image : np.ndarray | None
        Preloaded image array.
    output_image_path : str | None
        Path to save the sharpened image.

    Returns
    -------
    np.ndarray
        Sharpened image.
    """
    if not isinstance(coefficient, (int, float)) or coefficient < 0:
        raise ValueError("'coefficient' must be a non-negative number.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    blurred = apply_average_blur(np_image=np_image)

    mask = cv2.subtract(np_image, blurred)
    sharpened = cv2.addWeighted(np_image, 1 + coefficient, mask, -coefficient, 0)

    if output_image_path:
        print(IOHandler.save_image(sharpened, output_image_path))
    return sharpened
