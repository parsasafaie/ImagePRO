import sys
import time
import random
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to sys.path for local imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from human_analysis.face_analysis.face_detection import detect_faces
from blur import apply_median_blur
from sharpen import apply_laplacian_sharpening
from rotate import rotate_image_custom
from grayscale import convert_to_grayscale
from resize import resize_image

# Constants
DEFAULT_NUM_IMAGES = 200
DEFAULT_START_INDEX = 0
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_CAMERA_INDEX = 0
DEFAULT_DELAY = 0.1
DEFAULT_FACE_ID = "unknown"


def capture_bulk_pictures(
    folder_path: str | Path,
    face_id: str | int = DEFAULT_FACE_ID,
    num_images: int = DEFAULT_NUM_IMAGES,
    start_index: int = DEFAULT_START_INDEX,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    apply_blur: bool = False,
    apply_grayscale: bool = False,
    apply_sharpen: bool = False,
    apply_rotate: bool = False,
    apply_resize: tuple = False,
    delay: float = DEFAULT_DELAY,
) -> None:
    """
    Capture frames from webcam and save cropped face images, with optional preprocessing.

    Processing order (if enabled):
    median blur → laplacian sharpen → grayscale → resize → random rotate

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
    apply_blur : bool, default=False
        Apply median blur (filter_size=3) before other steps.
    apply_grayscale : bool, default=False
        Convert frame to single-channel grayscale.
    apply_sharpen : bool, default=False
        Apply Laplacian-based sharpening (coefficient=1.0).
    apply_rotate : bool, default=False
        Apply a random rotation in [-45°, +45°] with random scale {1.0, 1.1, 1.2, 1.3}.
    apply_resize : tuple[int, int] | None, default=None
        If provided, resize to (width, height) before rotation.
    delay : float, default=0.1
        Delay between each capture in seconds.

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
    if not isinstance(delay, (int, float)) or delay < 0:
        raise ValueError("'delay' must be a non-negative number.")

    base_dir = Path(folder_path)
    face_folder = base_dir / str(face_id)

    # Create destination folder (fail if exists to avoid accidental overwrite)
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

            # Optional preprocessing steps
            proc = frame

            if apply_blur:
                # Small kernel to keep facial details while reducing salt-and-pepper noise
                proc = apply_median_blur(src_np_image=proc, filter_size=3)

            if apply_sharpen:
                # Gentle sharpening; adjust coefficient if needed
                proc = apply_laplacian_sharpening(src_np_image=proc, coefficient=1.0)

            if apply_grayscale:
                proc = convert_to_grayscale(src_np_image=proc)

            if apply_resize is not False:
                proc = resize_image(new_size=apply_resize, src_np_image=proc)

            if apply_rotate:
                angle = float(random.randint(-45, 45))
                scale = random.choice([1.0, 1.1, 1.2, 1.3])
                proc = rotate_image_custom(src_np_image=proc, angle=angle, scale=scale)

            # Zero-padded filenames for better ordering
            filename = f"{start_index + saved:04d}.jpg"
            out_path = face_folder / filename

            try:
                # Save cropped face image via standardized detect_faces
                detect_faces(
                    max_faces=1,
                    min_confidence=min_confidence,
                    src_np_image=proc,
                    output_image_path=str(out_path),
                    face_mesh_obj=face_mesh,
                )
                saved += 1

                if delay > 0:
                    time.sleep(delay)

            except ValueError:
                # No face detected; skip this frame
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
        apply_blur=True,
        apply_sharpen=True,
        apply_grayscale=True,
        apply_resize=(224, 224),
        apply_rotate=True,
        delay=0.1
    )
