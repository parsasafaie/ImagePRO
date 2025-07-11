import cv2
import numpy as np
from blur import average_blur

def laplacian_filter(laplacian_coefficient=3, image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    laplacian = cv2.Laplacian(np_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    sharpen_image = np_image + laplacian_coefficient * laplacian
    sharpen_image = np.uint8(np.clip(sharpen_image, 0, 255))

    if result_path:
        cv2.imwrite(result_path, sharpen_image)
    else:
        return sharpen_image


def unsharp_masking(coefficient=1, image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    blur_image = average_blur(np_image=np_image)

    mask = cv2.subtract(np_image, blur_image)
    sharpen_image = cv2.addWeighted(np_image, 1 + coefficient, mask, -coefficient, 0)

    if result_path:
        cv2.imwrite(result_path, sharpen_image)
    else:
        return sharpen_image
