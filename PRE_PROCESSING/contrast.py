import cv2
from grayscale import grayscale

def enhance_contrast_clahe(clipLimit=2.0, tileGridSize=(8, 8), image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path,0)
    else:
        np_image = grayscale(np_image=np_image)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    enhanced_image = clahe.apply(np_image)

    if result_path:
        cv2.imwrite(result_path, enhanced_image)
    else:
        return enhanced_image


def enhance_contrast_GHE(image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path,0)
    else:
        np_image = grayscale(np_image=np_image)

    enhanced_image = cv2.equalizeHist(np_image)

    if result_path:
        cv2.imwrite(result_path, enhanced_image)
    else:
        return enhanced_image
