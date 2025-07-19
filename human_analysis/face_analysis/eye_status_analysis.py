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