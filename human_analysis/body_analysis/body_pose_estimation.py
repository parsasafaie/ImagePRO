import cv2
import mediapipe as mp
from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import new IOHandler
from io_handler import IOHandler

def detect_body_pose(model_accuracy=1, landmarks_idx=None, image_path=None, np_image=None, result_path=None):
    """
    Detects body landmarks using MediaPipe FaceMesh and visualizes or saves them.

    Parameters:
        landmarks_idx (list): List of landmark indices to extract/visualize. If None, uses all 33.
        image_path (str): Path to input image file. If provided, `np_image` will be ignored.
        np_image (np.ndarray): Pre-loaded image as NumPy array. Only used if `image_path` is None.
        result_path (str): Path to save output (image with landmarks or CSV). Supports `.jpg` and `.csv`.

    Returns:
        str | tuple(np.ndarray, list) | np.ndarray | list:
            - If `result_path` ends with `.jpg`: returns annotated image or saves it.
            - If `result_path` ends with `.csv`: returns list of coordinates or saves as CSV.
            - If no `result_path` is given: returns both annotated image and list of landmarks coordinates.

    Raises:
        ValueError: If inputs have invalid values or unsupported file extension.
    """
    # Validate specific parameters
    if landmarks_idx is not None and not isinstance(landmarks_idx, list):
        raise TypeError("'landmarks_idx' must be a list of integers or None.")

    if result_path is not None and not isinstance(result_path, str):
        raise TypeError("'result_path' must be a string or None.")

    if result_path and not (result_path.endswith('.jpg') or result_path.endswith('.csv')):
        raise ValueError("Only '.jpg' and '.csv' extensions are supported for 'result_path'.")

    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    bode_Pose = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=model_accuracy
    )

    # Load input image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Use default indices if none provided
    if landmarks_idx is None:
        landmarks_idx = list(range(33))
    
    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    result = bode_Pose.process(image_rgb)
    
    # Process detected body
    annotated_image = np_image.copy()
    all_landmarks = []

    if result.pose_landmarks:
        if result_path and result_path.endswith('.jpg') or result_path is None:
            if len(landmarks_idx) == 33:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_drawing.draw_landmarks(
                    annotated_image,
                    result.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            else:
                for idx in landmarks_idx:
                    ih, iw, _ = annotated_image.shape
                    landmark = result.pose_landmarks.landmark[idx]
                    x, y = int(iw * landmark.x), int(ih * landmark.y)
                    cv2.circle(annotated_image, (x, y), 3, (0, 0, 255), -1)

        if result_path and result_path.endswith('.csv') or result_path is None:
            for idx in landmarks_idx:
                landmark = result.pose_landmarks.landmark[idx]
                all_landmarks.append([idx, landmark.x, landmark.y, landmark.z])

    
    # Handle output
    if result_path:
        if result_path.endswith('.jpg'):
            return IOHandler.save_image(annotated_image, result_path)
        elif result_path.endswith('.csv'):
            return IOHandler.save_csv(all_landmarks, result_path)
    else:
        return annotated_image, all_landmarks


def detect_body_pose_live():
    """
    Live webcam capture with real-time body landmark detection and overlay.

    Raises:
        ValueError: If inputs are out of valid range.
        RuntimeError: If camera cannot be accessed or released.
    """
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

            try:
                landmarked_image = detect_body_pose(model_accuracy=1, np_image=image)[0]
            except ValueError:
                landmarked_image = image

            cv2.imshow('ImagePRO - Live Body pose detection', landmarked_image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    detect_body_pose_live()