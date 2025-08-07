import sys
from pathlib import Path

# Add parent directory to sys.path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler


def crop_image(start_point, end_point, image_path=None, np_image=None, result_path=None):
    """
    Crop image using top-left and bottom-right coordinates.

    Args:
        start_point (tuple): (x1, y1) top-left corner.
        end_point (tuple): (x2, y2) bottom-right corner.
        image_path (str, optional): Path to input image. Overrides `np_image` if provided.
        np_image (np.ndarray, optional): Image array. Used if `image_path` is None.
        result_path (str, optional): Path to save cropped image.

    Returns:
        str or np.ndarray: Save message if saved, else cropped image array.

    Raises:
        TypeError: If inputs are not proper tuples of integers.
        ValueError: If coordinates are invalid or out of image bounds.
    """
    # Validate coordinates
    if (
        not isinstance(start_point, tuple) or
        not isinstance(end_point, tuple) or
        len(start_point) != 2 or len(end_point) != 2 or
        not all(isinstance(c, int) for c in start_point + end_point)
    ):
        raise TypeError("Both 'start_point' and 'end_point' must be (x, y) tuples of integers.")

    x1, y1 = start_point
    x2, y2 = end_point

    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop coordinates: ensure (x1, y1) is top-left and (x2, y2) is bottom-right.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    height, width = np_image.shape[:2]

    if x2 > width or y2 > height:
        raise ValueError(f"Crop area exceeds image bounds ({width}x{height}).")

    cropped = np_image[y1:y2, x1:x2]
    return IOHandler.save_image(cropped, result_path)
