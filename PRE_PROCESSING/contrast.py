import cv2
from pathlib import Path
import sys

from grayscale import grayscale

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from image_manager import *

def enhance_contrast_clahe(clipLimit=2.0, tileGridSize=(8, 8), image_path=None, np_image=None, result_path=None):
    np_image = grayscale(np_image=input_manager(image_path, np_image))

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    enhanced_image = clahe.apply(np_image)

    return output_manger(enhanced_image, result_path)


def enhance_contrast_GHE(image_path=None, np_image=None, result_path=None):
    np_image = grayscale(np_image=input_manager(image_path, np_image))

    enhanced_image = cv2.equalizeHist(np_image)

    return output_manger(enhanced_image, result_path)


def contrast_stretching(alpha, beta, image_path=None, np_image=None, result_path=None):
    np_image = grayscale(np_image=input_manager(image_path, np_image))

    enhanced_image = cv2.convertScaleAbs(np_image, alpha=alpha, beta=beta)

    return output_manger(enhanced_image, result_path)
