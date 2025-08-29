import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to import custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
mp_pose = mp.solutions.pose
TOTAL_LANDMARKS = 33
DEFAULT_CONFIDENCE = 0.7
LANDMARK_RADIUS = 3
LANDMARK_COLOR = (0, 0, 255)  # Red color for landmarks

def detect_body_pose(
    *,
    image: Image,
    min_confidence: float = DEFAULT_CONFIDENCE,
    landmarks_idx: list | None = None,
    pose_obj=None
) -> Result:
    """
    Detect body landmarks from an image using MediaPipe Pose.

    Parameters
    ----------
    image : Image
        Image instance (BGR) to process.
    min_confidence : float, default=0.7
        Minimum detection confidence [0, 1].
    landmarks_idx : list[int] | None, optional
        Indices of landmarks to extract. Default: all 33.
    pose_obj : mp.solutions.pose.Pose | None, optional
        Optional pre-initialized pose model.

    Returns
    -------
    Result
        Result with `image` as annotated image and `data` as landmark rows [idx, x, y, z].
    """
    if not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")
    if landmarks_idx is not None and (not isinstance(landmarks_idx, list) or not all(isinstance(i, int) for i in landmarks_idx)):
        raise TypeError("'landmarks_idx' must be a list of ints or None.")

    h, w = image.shape[:2]
    np_image = image._data

    if landmarks_idx is None:
        landmarks_idx = list(range(TOTAL_LANDMARKS))

    if pose_obj is None:
        pose_obj = mp_pose.Pose(
            min_detection_confidence=min_confidence,
            static_image_mode=True
        )

    annotated_image = np_image.copy()

    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    result = pose_obj.process(image_rgb)

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
            for idx in landmarks_idx:
                lm = result.pose_landmarks.landmark[idx]
                x, y = int(w * lm.x), int(h * lm.y)
                cv2.circle(annotated_image, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

        for idx in landmarks_idx:
            lm = result.pose_landmarks.landmark[idx]
            all_landmarks.append([idx, lm.x, lm.y, lm.z])

    return Result(image=annotated_image, data=all_landmarks, meta={"source": image, "operation": "detect_body_pose", "min_confidence": min_confidence, "landmarks_idx": landmarks_idx})


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
                img = Image.from_array(frame)
                result = detect_body_pose(image=img, min_confidence=DEFAULT_CONFIDENCE, pose_obj=pose_obj)
                annotated_img = result.image if result.image is not None else frame
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
