import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to import custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler

# Constants
mp_pose = mp.solutions.pose
TOTAL_LANDMARKS = 33
DEFAULT_CONFIDENCE = 0.7
LANDMARK_RADIUS = 3
LANDMARK_COLOR = (0, 0, 255)  # Red color for landmarks

def detect_body_pose(
    model_accuracy: float = DEFAULT_CONFIDENCE, 
    landmarks_idx: list | None = None,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None,
    output_csv_path: str | None = None,
    pose_obj=None
):
    """
    Detect body landmarks from an image using MediaPipe Pose.

    Parameters
    ----------
    model_accuracy : float, default=0.7
        Minimum detection confidence [0, 1].
    landmarks_idx : list[int] | None, optional
        Indices of landmarks to extract. Default: all 33.
    src_image_path : str | None, optional
        Path to image file.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image array.
    output_image_path : str | None, optional
        If provided, saves annotated image.
    output_csv_path : str | None, optional
        If provided, saves landmarks CSV.
    pose_obj : mp.solutions.pose.Pose | None, optional
        Optional pre-initialized pose model.

    Returns
    -------
    tuple[np.ndarray, list]
        Annotated image and landmarks list.

    Raises
    ------
    ValueError, TypeError, FileNotFoundError, IOError
    """
    if landmarks_idx is not None and (not isinstance(landmarks_idx, list) or not all(isinstance(i, int) for i in landmarks_idx)):
        raise TypeError("'landmarks_idx' must be a list of ints or None.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)

    if landmarks_idx is None:
        landmarks_idx = list(range(TOTAL_LANDMARKS))

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
        if len(landmarks_idx) == TOTAL_LANDMARKS:
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
                cv2.circle(annotated_image, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

        for idx in landmarks_idx:
            lm = result.pose_landmarks.landmark[idx]
            all_landmarks.append([idx, lm.x, lm.y, lm.z])

    if output_image_path:
        print(IOHandler.save_image(annotated_image, output_image_path))
    if output_csv_path:
        print(IOHandler.save_csv(all_landmarks, output_csv_path))

    return annotated_image, all_landmarks


def detect_body_pose_live():
    """Starts webcam and shows real-time body pose detection. Press ESC to exit."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    pose_obj = mp_pose.Pose(
        min_detection_confidence=DEFAULT_CONFIDENCE,
        static_image_mode=False
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            try:
                annotated_img, _ = detect_body_pose(src_np_image=frame, pose_obj=pose_obj)
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
