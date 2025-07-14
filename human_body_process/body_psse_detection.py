import cv2
import mediapipe as mp
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler

def body_pose_detection(model_accuracy=1, landmarks_idx=None, image_path=None, np_image=None, result_path=None):
    body_pose = mp.solutions.pose.Pose(
        static_image_mode=False, 
        model_complexity=model_accuracy
    )

    # Load image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Use default indices if none provided
    if landmarks_idx is None:
        landmarks_idx = list(range(33))


    image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    result = body_pose.process(image_rgb)

    annotated_image = np_image.copy()
    ih, iw, _ = annotated_image.shape
    all_landmarks = []

    if result.pose_landmarks:
        for idx in landmarks_idx:
            if result_path and result_path.endswith('.jpg') or result_path is None:
                landmark = result.pose_landmarks.landmark[idx]
                x, y = int(iw * landmark.x), int(ih * landmark.y)
                cv2.circle(annotated_image, (x, y), 2, (0, 0, 0), -1)

            if result_path and result_path.endswith('.csv') or result_path is None:
                landmark = result.pose_landmarks.landmark[idx]
                all_landmarks.append([idx, landmark.x, landmark.y, landmark.z])

    if result_path:
        if result_path.endswith('.jpg'):
            return IOHandler.save_image(annotated_image, result_path)
        elif result_path.endswith('.csv'):
            return IOHandler.save_csv(all_landmarks, result_path)
    else:
        return annotated_image, all_landmarks
