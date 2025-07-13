from pathlib import Path
import sys

# Add parent directory to Python path for importing custom modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import custom modules
from io_handler import IOHandler
from human_face_process.face_mesh_analyzer import face_mesh


def head_pose_estimator(max_faces=1, min_confidence=0.7, image_path=None, np_image=None, result_path=None):
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
    _, landmarks = face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=important_indices,
        np_image=np_image
    )

    if not landmarks:
        raise ValueError("No face landmarks detected in the input image.")

    face_yaw_pitch = []

    # Process each detected face
    for face in landmarks:
        # Initialize landmarks dictionary
        landmark_dict = {idx: (x, y) for idx, x, y in face}

        try:
            nose_tip = landmark_dict[1]
            chin = landmark_dict[152]
            left_eye_outer = landmark_dict[33]
            right_eye_outer = landmark_dict[263]
            nasion = landmark_dict[168]
        except KeyError as e:
            raise ValueError(f"Missing expected landmark index: {e}")

        # Calculate yaw and pitch using simple proportional differences
        yaw = 100 * ((right_eye_outer[1] - nasion[1]) - (nasion[1] - left_eye_outer[1]))
        pitch = 100 * ((chin[1] - nose_tip[1]) - (nose_tip[1] - nasion[1]))

        face_yaw_pitch.append([face[0][0], yaw, pitch])  # face ID, yaw, pitch

    # Save or return result
    if result_path:
        return IOHandler.save_csv(face_yaw_pitch, result_path)
    else:
        return face_yaw_pitch
    