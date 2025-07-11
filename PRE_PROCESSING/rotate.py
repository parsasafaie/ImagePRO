import cv2

def rotate_90(image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    rotated_image = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)

    if result_path:
        cv2.imwrite(result_path, rotated_image)
    else:
        return rotated_image
    

def rotate_180(image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    rotated_image = cv2.rotate(np_image, cv2.ROTATE_180)

    if result_path:
        cv2.imwrite(result_path, rotated_image)
    else:
        return rotated_image
    

def rotate_270(image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)

    rotated_image = cv2.rotate(np_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if result_path:
        cv2.imwrite(result_path, rotated_image)
    else:
        return rotated_image


def rotate(angle, scale=1.0, image_path=None, np_image=None, result_path=None):
    if image_path:
        np_image = cv2.imread(image_path)
     
    height, width = np_image.shape[:2]
    image_center = (width/2, height/2)

    rotation_matrix = cv2.getRotationMatrix2D(center=image_center, angle=angle, scale=scale)
    rotated_image = cv2.warpAffine(np_image, rotation_matrix, (width, height))

    if result_path:
        cv2.imwrite(result_path, rotated_image)
    else:
        return rotated_image
