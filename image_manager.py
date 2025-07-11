import cv2

def input_manager(image_path=None, np_image=None):
    np_image = cv2.imread(image_path) if image_path else np_image
    return np_image

def output_manger(np_image, result_path=None):
    if result_path:
        cv2.imwrite(result_path, np_image)
        return f"File saved at {result_path}"
    else:
        return np_image
    