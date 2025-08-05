from pathlib import Path
import sys
import cv2

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import custom modules
from io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh


def analyze_eye_status(min_confidence=0.7, image_path=None, np_image=None):
    """
    Detects eye status (open or closed) from facial landmarks in an image.
    
    Parameters:
        min_confidence (float): Minimum detection confidence threshold.
    
    Raises:
        ValueError: If invalid parameters are provided.
        RuntimeError: If webcam cannot be opened.
    """
    # Load input image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Define important landmark indices
    important_indices = [386, 374, 263, 362]

    # Detect facial landmarks
    _, landmarks = analyze_face_mesh(
        max_faces=1,
        min_confidence=min_confidence,
        landmarks_idx=important_indices,
        np_image=np_image
    )
    
    if not landmarks:
        raise ValueError("No face landmarks detected in the input image.")

    for landmark in landmarks[0]:
            idx = landmark[1]
            if idx==263:
                outer_corner = landmark[2]
            elif idx==362:
                inner_corner = landmark[2]
            elif idx==374:
                bottom = landmark[3]
            elif idx==386:
                top = landmark[3]
    
    # Calculate the vertical and horizontal distances
    vertical_dist = bottom - top
    horizontal_dist = outer_corner - inner_corner

    # Compute the Eye Aspect Ratio (EAR)
    ear = vertical_dist / horizontal_dist

    return ear > 0.2

def analyze_eye_status_live(min_confidence=0.7):
    """
    
    Analyze eye status in real-time using webcam input.
    
    Parameters:
        min_confidence (float): Minimum detection confidence threshold.
    
    Raises:
        ValueError: If invalid parameters are provided.
        RuntimeError: If webcam cannot be opened.
    """
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

            # Detect eye status
            eye_status = analyze_eye_status(
                image_path=None,
                min_confidence=min_confidence,
                np_image=image
            )

            text = f"{eye_status}"
            cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Show the resulting frame
            cv2.imshow('ImagePRO - Live Eye Status Analyzer', image)

            # Exit on ESC key press
            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    analyze_eye_status_live()