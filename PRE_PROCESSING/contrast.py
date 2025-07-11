import cv2
from pathlib import Path
import sys

# Import grayscale utility from your custom module
from grayscale import grayscale

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import input/output managers from your custom module
from image_manager import input_manager, output_manger


def enhance_contrast_clahe(clipLimit=2.0, tileGridSize=(8, 8), image_path=None, np_image=None, result_path=None):
    """
    Enhances image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Parameters:
        clipLimit (float): Threshold for contrast limiting (higher = more contrast).
        tileGridSize (tuple): Size of grid for histogram equalization (smaller = finer details).
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the enhanced image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the enhanced image as a NumPy array.

    Note:
        At least one of `image_path` or `np_image` must be provided.
        Input image is automatically converted to grayscale before enhancement.
    """
    # Load and convert image to grayscale
    np_image = grayscale(np_image=input_manager(image_path=image_path, np_image=np_image))

    # Create CLAHE object with specified parameters
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    # Apply CLAHE to enhance contrast
    enhanced_image = clahe.apply(np_image)

    # Output the result (save or return)
    return output_manger(enhanced_image, result_path)


def enhance_contrast_GHE(image_path=None, np_image=None, result_path=None):
    """
    Enhances image contrast using Global Histogram Equalization (GHE).
    
    Parameters:
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the enhanced image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the enhanced image as a NumPy array.

    Note:
        At least one of `image_path` or `np_image` must be provided.
        Input image is automatically converted to grayscale before enhancement.
    """
    # Load and convert image to grayscale
    np_image = grayscale(np_image=input_manager(image_path=image_path, np_image=np_image))

    # Apply global histogram equalization
    enhanced_image = cv2.equalizeHist(np_image)

    # Output the result (save or return)
    return output_manger(enhanced_image, result_path)


def contrast_stretching(alpha, beta, image_path=None, np_image=None, result_path=None):
    """
    Enhances image contrast using linear stretching with alpha (contrast) and beta (brightness).
    
    Parameters:
        alpha (float): Contrast scaling factor (1.0 means no change).
        beta (int): Brightness adjustment value (0 means no change).
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save the enhanced image (optional). If not provided, returns the image array.

    Returns:
        str | np.ndarray: If `result_path` is given, returns confirmation message.
                          Otherwise, returns the enhanced image as a NumPy array.

    Note:
        At least one of `image_path` or `np_image` must be provided.
        Input image is automatically converted to grayscale before enhancement.
    """
    # Load and convert image to grayscale
    np_image = grayscale(np_image=input_manager(image_path=image_path, np_image=np_image))

    # Apply contrast stretching using alpha and beta
    enhanced_image = cv2.convertScaleAbs(np_image, alpha=alpha, beta=beta)

    # Output the result (save or return)
    return output_manger(enhanced_image, result_path)