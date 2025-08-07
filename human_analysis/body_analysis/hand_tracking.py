import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to import custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler

mp_hands = mp.solutions.hands


def detect_hands(
    max_hands=2,
    min_confidence=0.7,
    landmarks_idx=None,
    image_path=None,
    np_image=None,
    result_path=None,
    hands_obj=None
):
    """
    Detects hand landmarks in an image using MediaPipe.

    Args:
        max_hands (int): Max number of hands to detect.
        min_confidence (float): Minimum confidence for detection.
        landmarks_idx (list[int] | None): Landmark indices to extract (default: all 21).
        image_path (str | None): Image file path.
        np_image (np.ndarray | None): Preloaded image array.
        result_path (str | None): Path to save output image (.jpg) or landmarks (.csv).
        hands_obj (mp.solutions.hands.Hands | None): Optional pre-initialized model.

    Returns:
        np.ndarray | list: Annotated image and landmarks list, or saved file confirmation.
    """
    if not isinstance(max_hands, int) or max_hands <= 0:
        raise ValueError("'max_hands' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")

    if landmarks_idx is not None and not isinstance(landmarks_idx, list):
        raise TypeError("'landmarks_idx' must be a list or None.")

    if result_path:
        if not isinstance(result_path, str):
            raise TypeError("'result_path' must be a string.")
        if not (result_path.endswith('.jpg') or result_path.endswith('.csv')):
            raise ValueError("Only '.jpg' and '.csv' extensions are supported.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    if hands_obj is None:
        hands_obj = mp_hands.Hands(
            min_detection_confidence=min_confidence,
            max_num_hands=max_hands,
            static_image_mode=True
        )

    if landmarks_idx is None:
        landmarks_idx = list(range(21))

    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    results = hands_obj.process(rgb_image)

    annotated_image = np_image.copy()
    all_landmarks = []

    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if result_path is None or result_path.endswith('.jpg'):
                if len(landmarks_idx) == 21:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=annotated_image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
                else:
                    h, w, _ = annotated_image.shape
                    for idx in landmarks_idx:
                        lm = hand_landmarks.landmark[idx]
                        x, y = int(w * lm.x), int(h * lm.y)
                        cv2.circle(annotated_image, (x, y), 3, (0, 0, 255), -1)

            if result_path is None or result_path.endswith('.csv'):
                hand_data = []
                for idx in landmarks_idx:
                    lm = hand_landmarks.landmark[idx]
                    hand_data.append([hand_id, idx, lm.x, lm.y, lm.z])
                all_landmarks.append(hand_data)
    else:
        return np_image, []

    if result_path:
        if result_path.endswith('.jpg'):
            return IOHandler.save_image(annotated_image, result_path)
        elif result_path.endswith('.csv'):
            flat_landmarks = [item for sublist in all_landmarks for item in sublist]
            return IOHandler.save_csv(flat_landmarks, result_path)
    else:
        return annotated_image, all_landmarks


def detect_hands_live(max_hands=2, min_confidence=0.7):
    """
    Real-time hand detection via webcam.

    Args:
        max_hands (int): Maximum hands to detect.
        min_confidence (float): Detection confidence threshold.
    """
    if not isinstance(max_hands, int) or max_hands <= 0:
        raise ValueError("'max_hands' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    hands_obj = mp_hands.Hands(
        min_detection_confidence=min_confidence,
        max_num_hands=max_hands,
        static_image_mode=True
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            annotated_img, _ = detect_hands(
                max_hands=max_hands,
                min_confidence=min_confidence,
                np_image=frame,
                hands_obj=hands_obj
            )

            cv2.imshow('Live hand detector - ImagePRO', annotated_img)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_hands_live(max_hands=2)
