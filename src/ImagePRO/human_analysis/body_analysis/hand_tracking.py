import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to import custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler

# Constants
mp_hands = mp.solutions.hands
DEFAULT_MAX_HANDS = 2
DEFAULT_MIN_CONFIDENCE = 0.7
TOTAL_HAND_LANDMARKS = 21
LANDMARK_RADIUS = 3
LANDMARK_COLOR = (0, 0, 255)  # Red color for landmarks


def detect_hands(
    max_hands: int = DEFAULT_MAX_HANDS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    landmarks_idx: list | None = None,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None,
    output_csv_path: str | None = None,
    hands_obj=None
):
    """
    Detect hand landmarks in an image using MediaPipe Hands.

    Parameters
    ----------
    max_hands : int, default=2
        Max number of hands to detect.
    min_confidence : float, default=0.7
        Minimum detection confidence [0, 1].
    landmarks_idx : list[int] | None, optional
        Landmark indices to extract (default: all 21).
    src_image_path : str | None, optional
        Path to image file.
    src_np_image : np.ndarray | None, optional
        Preloaded BGR image array.
    output_image_path : str | None, optional
        If provided, saves annotated image.
    output_csv_path : str | None, optional
        If provided, saves landmarks CSV.
    hands_obj : mp.solutions.hands.Hands | None, optional
        Optional pre-initialized model.

    Returns
    -------
    tuple[np.ndarray, list]
        Annotated image and landmarks list.

    Raises
    ------
    ValueError, TypeError, FileNotFoundError, IOError
    """
    if not isinstance(max_hands, int) or max_hands <= 0:
        raise ValueError("'max_hands' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")
    if landmarks_idx is not None and (not isinstance(landmarks_idx, list) or not all(isinstance(i, int) for i in landmarks_idx)):
        raise TypeError("'landmarks_idx' must be a list of ints or None.")

    np_image = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)

    if hands_obj is None:
        hands_obj = mp_hands.Hands(
            min_detection_confidence=min_confidence,
            max_num_hands=max_hands,
            static_image_mode=True
        )

    if landmarks_idx is None:
        landmarks_idx = list(range(TOTAL_HAND_LANDMARKS))

    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    results = hands_obj.process(rgb_image)

    annotated_image = np_image.copy()
    all_landmarks = []

    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if len(landmarks_idx) == TOTAL_HAND_LANDMARKS:
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
                    cv2.circle(annotated_image, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

            for idx in landmarks_idx:
                lm = hand_landmarks.landmark[idx]
                all_landmarks.append([hand_id, idx, lm.x, lm.y, lm.z])

    if output_image_path:
        print(IOHandler.save_image(annotated_image, output_image_path))
    if output_csv_path:
        print(IOHandler.save_csv(all_landmarks, output_csv_path))

    return annotated_image, all_landmarks


def detect_hands_live(max_hands: int = DEFAULT_MAX_HANDS, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
    """Real-time hand detection via webcam. Press ESC to exit."""
    if not isinstance(max_hands, int) or max_hands <= 0:
        raise ValueError("'max_hands' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    hands_obj = mp_hands.Hands(
        min_detection_confidence=min_confidence,
        max_num_hands=max_hands,
        static_image_mode=False
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
                src_np_image=frame,
                hands_obj=hands_obj
            )

            cv2.imshow('Live hand detector - ImagePRO', annotated_img)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_hands_live(max_hands=DEFAULT_MAX_HANDS)
