import cv2
import numpy as np

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
