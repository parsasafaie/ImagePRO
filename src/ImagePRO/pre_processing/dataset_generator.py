import sys
from pathlib import Path
import cv2
import mediapipe as mp

# Add parent directory to sys.path for local imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from human_analysis.face_analysis.face_detection import detect_faces  # standardized detect_faces


def capture_bulk_pictures(
    folder_path: str | Path,
    face_id: str | int,
    num_images: int = 200,
    start_index: int = 0,
    min_confidence: float = 0.7,
    camera_index: int = 0,
) -> None:
    """
    Capture frames from webcam and save cropped face images.

    Parameters
    ----------
    folder_path : str | Path
        Base directory where the face-id folder will be created.
    face_id : str | int
        Subfolder name (e.g., user id). A new folder "<folder_path>/<face_id>" will be created.
    num_images : int, default=200
        Number of frames to capture/save.
    start_index : int, default=0
        Starting index for saved filenames (zero-padded).
    min_confidence : float, default=0.7
        Detection confidence for MediaPipe FaceMesh.
    camera_index : int, default=0
        OpenCV camera index.

    Raises
    ------
    ValueError
        If arguments are invalid.
    FileExistsError
        If the destination folder already exists.
    RuntimeError
        If the webcam cannot be opened.
    """
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("'num_images' must be a positive integer.")
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("'start_index' must be a non-negative integer.")
    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    base_dir = Path(folder_path)
    face_folder = base_dir / str(face_id)

    # Create destination folder
    try:
        face_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        raise FileExistsError(f"Destination already exists: {face_folder}") from e

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam (index={camera_index}).")

    # Initialize FaceMesh for live stream (tracking mode)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=False,
    )

    saved = 0
    try:
        while saved < num_images:
            ok, frame = cap.read()
            if not ok:
                print("Skipping empty frame.")
                continue

            # Zero-padded filenames for better ordering
            filename = f"{start_index + saved:04d}.jpg"
            out_path = face_folder / filename

            try:
                detect_faces(
                    max_faces=1,
                    min_confidence=min_confidence,
                    src_np_image=frame,
                    output_image_path=str(out_path),
                    face_mesh_obj=face_mesh,
                )
                saved += 1
            except ValueError:
                # No face detected; skip frame
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_bulk_pictures(
        folder_path=r"tmp",
        face_id="0",
        num_images=200,
        start_index=0,
        min_confidence=0.7,
        camera_index=0,
    )
