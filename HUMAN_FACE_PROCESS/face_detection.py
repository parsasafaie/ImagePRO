import cv2
from face_mesh import face_mesh
from pathlib import Path
import sys
import numpy as np

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler

def face_extraction(image_path=None, np_image=None, result_path=None):
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)
    h, w, _ = np_image.shape

    face_outline_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
        176, 149, 150, 136, 164, 163, 153, 157
    ]
    landmarks = face_mesh(landmarks_idx=face_outline_indices, image_path=image_path, np_image=np_image)[1]
    
    for face_index, face in enumerate(landmarks):
        for landmark_index, landmark in enumerate(face):
            print(landmark)
            landmarks[face_index][landmark_index] = [[landmark[2]*w, landmark[3]*h]]

    for i, face in enumerate(landmarks):
        landmarks[i] = np.array(face, dtype=np.int32)
    

    for face in landmarks:
        x, y, w, h = cv2.boundingRect(face)
        cropped_face = np_image[y:y+h, x:x+w]

    return IOHandler.save_image(cropped_face, result_path=result_path)
    