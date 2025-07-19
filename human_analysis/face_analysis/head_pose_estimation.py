from pathlib import Path
import sys
import cv2

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import custom modules
from io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh


def estimate_head_pose(max_faces=1, min_confidence=0.7, image_path=None, np_image=None, result_path=None):
    """
    Estimates the head pose (yaw and pitch) from facial landmarks in an image.

    Parameters:
        max_faces (int): Maximum number of faces to detect.
        min_confidence (float): Minimum detection confidence threshold.
        image_path (str | None): Path to input image file.
        np_image (np.ndarray | None): Pre-loaded image array.
        result_path (str | None): Path to save the output CSV results.

    Returns:
        str | list[list]: CSV save message if result_path is provided, else list of [face_id, yaw, pitch].
    """

    # Validate specific parameters
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
        raise ValueError("'min_confidence' must be a float between 0.0 and 1.0.")

    # Load input image
    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Define important landmark indices
    important_indices = [1, 152, 33, 263, 168]

    # Detect facial landmarks
    _, landmarks = analyze_face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=important_indices,
        np_image=np_image
    )

    if not landmarks:
        raise ValueError("No face landmarks detected in the input image.")

    face_yaw_pitch = []
    for face in landmarks:
        for landmark in face:
            idx = landmark[1]
            if idx==1:
                nose_tip = landmark
            elif idx==152:
                chin = landmark
            elif idx==33:
                left_eye_outer = landmark
            elif idx==263:
                right_eye_outer = landmark
            elif idx==168:
                nasion = landmark

        # Calculate yaw and pitch using simple proportional differences
        yaw = 100 * ((right_eye_outer[2] - nasion[2]) - (nasion[2] - left_eye_outer[2]))
        pitch = 100 * ((chin[3] - nose_tip[3]) - (nose_tip[3] - nasion[3]))

        face_yaw_pitch.append([face[0][0], yaw, pitch])  # face ID, yaw, pitch

    # Save or return result
    if result_path:
        return IOHandler.save_csv(face_yaw_pitch, result_path)
    else:
        return face_yaw_pitch


def estimate_head_pose_live(max_faces=1, min_confidence=0.7):
    """
    Estimates head pose (yaw and pitch) in real-time using webcam input.
    
    Parameters:
        max_faces (int): Maximum number of faces to detect.
        min_confidence (float): Minimum detection confidence threshold.
    
    Raises:
        ValueError: If invalid parameters are provided.
        RuntimeError: If webcam cannot be opened.
    """

    # Validate specific parameters
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

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

            # Estimate head pose for each detected face
            face_yaw_pitch = estimate_head_pose(
                max_faces=max_faces,
                min_confidence=min_confidence,
                np_image=image
            )

            # Overlay yaw and pitch info on the image
            first_text_h = 50
            for i, face in enumerate(face_yaw_pitch):
                text = f"Face {int(face[0])}: Yaw={face[1]:.2f}, Pitch={face[2]:.2f}"
                cv2.putText(image, text, (10, first_text_h + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Show the resulting frame
            cv2.imshow('ImagePRO - Live Head Pose Estimator', image)

            # Exit on ESC key press
            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()