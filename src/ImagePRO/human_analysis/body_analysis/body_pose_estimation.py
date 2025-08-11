import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to import custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler

mp_pose = mp.solutions.pose

def detect_body_pose(
    model_accuracy=0.7,
    landmarks_idx=None,
    image_path=None,
    np_image=None,
    result_path=None,
    pose_obj=None
):
    """
    Detects body landmarks from an image using MediaPipe Pose.

    Args:
        model_accuracy (float): Confidence threshold (0.0 to 1.0).
        landmarks_idx (list[int] | None): Indices to extract (default: all 33).
        image_path (str | None): Path to image file.
        np_image (np.ndarray | None): Already loaded image (used if image_path is None).
        result_path (str | None): Path to save image (.jpg) or coordinates (.csv).
        pose_obj (mp.solutions.pose.Pose | None): Optional pre-initialized pose model.

    Returns:
        str | tuple[np.ndarray, list]: Output message or annotated image and landmark list.
    """
    if landmarks_idx is not None and not isinstance(landmarks_idx, list):
        raise TypeError("'landmarks_idx' must be a list or None.")

    if result_path:
        if not isinstance(result_path, str):
            raise TypeError("'result_path' must be a string.")
        if not (result_path.endswith('.jpg') or result_path.endswith('.csv')):
            raise ValueError("Supported extensions: '.jpg', '.csv'.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    if landmarks_idx is None:
        landmarks_idx = list(range(33))

    if pose_obj is None:
        pose_obj = mp_pose.Pose(
            min_detection_confidence=model_accuracy,
            static_image_mode=True
        )

    image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    result = pose_obj.process(image_rgb)

    annotated_image = np_image.copy()
    all_landmarks = []

    if result.pose_landmarks:
        if result_path is None or result_path.endswith('.jpg'):
            if len(landmarks_idx) == 33:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
            else:
                h, w, _ = annotated_image.shape
                for idx in landmarks_idx:
                    lm = result.pose_landmarks.landmark[idx]
                    x, y = int(w * lm.x), int(h * lm.y)
                    cv2.circle(annotated_image, (x, y), 3, (0, 0, 255), -1)

        if result_path is None or result_path.endswith('.csv'):
            for idx in landmarks_idx:
                lm = result.pose_landmarks.landmark[idx]
                all_landmarks.append([idx, lm.x, lm.y, lm.z])

    if result_path:
        if result_path.endswith('.jpg'):
            return IOHandler.save_image(annotated_image, result_path)
        elif result_path.endswith('.csv'):
            return IOHandler.save_csv(all_landmarks, result_path)
    else:
        return annotated_image, all_landmarks


def detect_body_pose_live():
    """
    Starts webcam and shows real-time body pose detection.
    Press ESC to exit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    pose_obj = mp_pose.Pose(
        min_detection_confidence=0.7,
        static_image_mode=True
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            try:
                annotated_img, _ = detect_body_pose(np_image=frame, pose_obj=pose_obj)
            except ValueError:
                annotated_img = frame

            cv2.imshow('ImagePRO - Live Body Pose Detection', annotated_img)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_body_pose_live()
