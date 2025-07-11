import cv2
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from image_manager import *

def rotate_90(image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    rotated_image = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)

    return output_manger(rotated_image, result_path)
    

def rotate_180(image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    rotated_image = cv2.rotate(np_image, cv2.ROTATE_180)

    return output_manger(rotated_image, result_path)
    

def rotate_270(image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    rotated_image = cv2.rotate(np_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return output_manger(rotated_image, result_path)


def rotate(angle, scale=1.0, image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)
     
    height, width = np_image.shape[:2]
    image_center = (width/2, height/2)

    rotation_matrix = cv2.getRotationMatrix2D(center=image_center, angle=angle, scale=scale)
    rotated_image = cv2.warpAffine(np_image, rotation_matrix, (width, height))

    return output_manger(rotated_image, result_path)
