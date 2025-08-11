import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to sys.path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from io_handler import IOHandler
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

mp_face_mesh = mp.solutions.face_mesh


def estimate_head_pose(
    max_faces=1,
    min_confidence=0.7,
    image_path=None,
    np_image=None,
    result_path=None,
    face_mesh_obj=None
):
    """
    Estimate head pose (yaw, pitch) from facial landmarks.

    Args:
        max_faces (int): Number of faces to detect.
        min_confidence (float): Confidence threshold (0–1).
        image_path (str, optional): Path to input image.
        np_image (np.ndarray, optional): Input image as array.
        result_path (str, optional): If given, save results as CSV.
        face_mesh_obj (FaceMesh, optional): Reusable MediaPipe instance.

    Returns:
        str or list[list]: CSV confirmation if saved, else list of [face_id, yaw, pitch].

    Raises:
        ValueError: On invalid input or no detected face.
    """
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    np_image = IOHandler.load_image(image_path=image_path, np_image=np_image)

    # Important landmark indices (MediaPipe 468 model)
    indices = [1, 152, 33, 263, 168]  # nose_tip, chin, left_eye, right_eye, nasion

    _, landmarks = analyze_face_mesh(
        max_faces=max_faces,
        min_confidence=min_confidence,
        landmarks_idx=indices,
        np_image=np_image,
        face_mesh_obj=face_mesh_obj
    )

    if not landmarks:
        raise ValueError("No face landmarks detected.")

    results = []
    for face in landmarks:
        points = {lm[1]: lm for lm in face}

        try:
            nose_x, nose_y = points[1][2:4]
            chin_y = points[152][3]
            left_x = points[33][2]
            right_x = points[263][2]
            nasion_x, nasion_y = points[168][2:4]
        except KeyError:
            continue  # skip this face if any point is missing

        # Simplified proportional estimation
        yaw = 100 * ((right_x - nasion_x) - (nasion_x - left_x))
        pitch = 100 * ((chin_y - nose_y) - (nose_y - nasion_y))

        results.append([face[0][0], yaw, pitch])  # face_id, yaw, pitch

    if result_path:
        return IOHandler.save_csv(results, result_path)

    return results


def estimate_head_pose_live(max_faces=1, min_confidence=0.7):
    """
    Live head pose estimation using webcam.

    Args:
        max_faces (int): Number of faces to detect.
        min_confidence (float): Confidence threshold (0–1).

    Raises:
        ValueError, RuntimeError: On invalid inputs or camera failure.
    """
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to access webcam.")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=True
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Skipping empty frame.")
                continue

            try:
                face_angles = estimate_head_pose(
                    max_faces=max_faces,
                    min_confidence=min_confidence,
                    np_image=frame,
                    face_mesh_obj=face_mesh
                )
            except ValueError:
                face_angles = []

            for i, face in enumerate(face_angles):
                face_id, yaw, pitch = face
                text = f"Face {int(face_id)+1}: Yaw={yaw:.2f}, Pitch={pitch:.2f}"
                cv2.putText(frame, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("ImagePRO - Head Pose Estimation", frame)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    estimate_head_pose_live()
