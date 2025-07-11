import cv2

def average_blur(kernel_size=(5, 5), image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    blur_image = cv2.blur(np_image, kernel_size)

    if result_path:
        cv2.imwrite(result_path, blur_image)
    else:
        return blur_image
