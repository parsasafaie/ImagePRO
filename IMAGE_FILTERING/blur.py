import cv2
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from image_manager import *

def average_blur(kernel_size=(5, 5), image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    blurred_image = cv2.blur(np_image, kernel_size)

    return output_manger(blurred_image, result_path)


def gaussian_blur(kernel_size=(5, 5), image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    blurred_image = cv2.GaussianBlur(np_image, kernel_size, 0)

    return output_manger(blurred_image, result_path)
    

def median_blur(filter_size=5, image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    blurred_image = cv2.medianBlur(np_image, filter_size)

    return output_manger(blurred_image, result_path)


def bilateral_blur(filter_size=9, sigma_color=75, sigma_space=75, image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path, np_image)

    blurred_image = cv2.bilateralFilter(np_image, d=filter_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    return output_manger(blurred_image, result_path)
