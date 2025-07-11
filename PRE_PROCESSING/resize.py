import cv2
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from image_manager import *

def resize(new_size, image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    resized_image = cv2.resize(np_image, dsize=new_size)

    return output_manger(resized_image, result_path)
    