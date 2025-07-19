import cv2
import mediapipe as mp
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler


def detect_hands(max_hands=2, min_confidence=0.7, landmarks_idx=None, image_path=None, np_image=None, result_path=None):
    # Validate specific parameters
    if not isinstance(max_hands, int) or max_hands <= 0:
        raise ValueError("'max_hands' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")

    if landmarks_idx is not None and not isinstance(landmarks_idx, list):
        raise TypeError("'landmarks_idx' must be a list of integers or None.")

    if result_path is not None and not isinstance(result_path, str):
        raise TypeError("'result_path' must be a string or None.")

    if result_path and not (result_path.endswith('.jpg') or result_path.endswith('.csv')):
        raise ValueError("Only '.jpg' and '.csv' extensions are supported for 'result_path'.")

    # Load input image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Initialize MediaPipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,      
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence
    )

    mp_drawing_utils = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Convert image to RGB for MediaPipe
    rgb_color = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_color)

    # Use default indices if none provided
    if landmarks_idx is None:
        landmarks_idx = list(range(21))

    # Process each detected hand
    annotated_image = np_image.copy()
    all_landmarks = []
    
    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if  result_path and result_path.endswith('.jpg') or result_path is None:
                if len(landmarks_idx) == 21:
                    mp_drawing_utils.draw_landmarks(
                        image=annotated_image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )
                else:
                    for idx in landmarks_idx:
                        ih, iw, _ = annotated_image.shape
                        landmark = hand_landmarks.landmark[idx]
                        x, y = int(iw * landmark.x), int(ih * landmark.y)
                        cv2.circle(annotated_image, (x, y), 3, (0, 0, 255), -1)

            if result_path and result_path.endswith('.csv') or result_path is None:
                landmarks_list = []
                for idx in landmarks_idx:
                    landmark = hand_landmarks.landmark[idx]
                    landmarks_list.append([hand_id, idx, landmark.x, landmark.y, landmark.z])
                all_landmarks.append(landmarks_list)
    else:
        return np_image,[]

    # Handle output
    if result_path:
        if result_path.endswith('.jpg'):
            return IOHandler.save_image(annotated_image, result_path)
        elif result_path.endswith('.csv'):
            # Flatten list if multiple hands
            flat_landmarks = [item for sublist in all_landmarks for item in sublist]
            return IOHandler.save_csv(flat_landmarks, result_path)
    else:
        return annotated_image, all_landmarks


def detect_hands_live(max_hands=2, min_confidence=0.7):
    # Validate specific parameters
    if not isinstance(max_hands, int) or max_hands <= 0:
        raise ValueError("'max_hands' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")
    
    try:
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Detect hands in image
            result = detect_hands(max_hands=max_hands, min_confidence=min_confidence, np_image=image)[0]

            # Show the resulting frame
            cv2.imshow('Live hand detector - ImagePRO', result)

            # Exit on ESC key press
            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
