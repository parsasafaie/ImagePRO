import cv2

def resize(new_size, image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    resized_image = cv2.resize(np_image, dsize=new_size)

    if result_path:
        cv2.imwrite(result_path, resized_image)
    else:
        return resized_image
    