import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler
from human_face_process.face_mesh_analyzer import face_mesh

def head_pose_estimator(max_faces=1, min_confidence=0.7, image_path=None, np_image=None, result_path=None):

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    important_indices = [1, 152, 33, 263, 168]

    landmarks = face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=important_indices,
        np_image=np_image
    )[1]

    if not landmarks:
        raise ValueError("No face landmarks detected in the input image.")
    
    face_yaw_pitch = []
    for face in landmarks:
        for landmark in face:
            idx = landmark[1]
            if idx==1:
                nose_tip_landmark = landmark
            elif idx==152:
                chin_landmark = landmark
            elif idx==33:
                left_eye_outer_landmark = landmark
            elif idx==263:
                right_eye_outer_landmark = landmark
            elif idx==168:
                nasion_landmark = landmark

        yaw = 100 * ((right_eye_outer_landmark[2]-nasion_landmark[2]) - (nasion_landmark[2]-left_eye_outer_landmark[2]))
        pitch = 100 * (chin_landmark[3]-nose_tip_landmark[3]) - (nose_tip_landmark[3]-nasion_landmark[3])

        face_yaw_pitch.append([nose_tip_landmark[0], yaw, pitch])

    if result_path:
        return IOHandler.save_csv(face_yaw_pitch, result_path)
    else:
        return face_yaw_pitch
    