import cv2
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from image_manager import *

def grayscale(image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    grayscale_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    return output_manger(grayscale_image, result_path)
