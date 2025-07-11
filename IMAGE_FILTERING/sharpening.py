import cv2
import numpy as np
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from image_manager import *
from blur import average_blur

def laplacian_filter(laplacian_coefficient=3, image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    laplacian = cv2.Laplacian(np_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    sharpen_image = np_image + laplacian_coefficient * laplacian
    sharpen_image = np.uint8(np.clip(sharpen_image, 0, 255))

    return output_manger(sharpen_image, result_path)


def unsharp_masking(coefficient=1, image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    blur_image = average_blur(np_image=np_image)

    mask = cv2.subtract(np_image, blur_image)
    sharpen_image = cv2.addWeighted(np_image, 1 + coefficient, mask, -coefficient, 0)

    return output_manger(sharpen_image, result_path)
