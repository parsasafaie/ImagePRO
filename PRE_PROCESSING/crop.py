import cv2
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from image_manager import *

def crop(x1, x2, y1, y2, image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    cropped_image = np_image[y1:y2, x1:x2]

    return output_manger(cropped_image, result_path)
