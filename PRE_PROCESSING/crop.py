import cv2

def crop(x1, x2, y1, y2, image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    croped_image = np_image[y1:y2, x1:x2]

    if result_path:
        cv2.imwrite(result_path, croped_image)
    else:
        return croped_image