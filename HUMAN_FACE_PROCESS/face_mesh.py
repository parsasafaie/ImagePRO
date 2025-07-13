import mediapipe as mp
import cv2
import numpy as np
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from image_manager import input_manager

def face_mesh(max_faces=1, min_confidence=0.7, landmarks_idx=None, image_path=None, np_image=None, result_path=None):
    np_image = input_manager(image_path=image_path, np_image=np_image)

    FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=True
    )

    rgb_color = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    results = FACE_MESH.process(rgb_color)

    if results.multi_face_landmarks:
        if not isinstance(landmarks_idx, list):
            landmarks_idx = [i for i in range(468)]

        for face_landmarks in results.multi_face_landmarks:
            if result_path.endswith('.jpg'):
                for idx in landmarks_idx:
                    landmark = face_landmarks[idx]
                    ih, iw, _ = np_image.shape
                    x, y = int(iw*landmark.x), int(ih*landmark.y)
                    cv2.circle(np_image, (x, y), 1, (0, 255, 0), -1)

                cv2.imwrite(np_image, result_path)

            elif result_path.endswith('.csv'):
                landmarks_list = []
                for idx in landmarks_idx:
                    landmarks_list.append([face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y, face_landmarks.landmark[idx].z])
                np.savetext(result_path, landmarks_list)
