import cv2

def grayscale(image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    grayscale_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    if result_path:
        cv2.imwrite(result_path, grayscale_image)
    else:
        return grayscale_image
