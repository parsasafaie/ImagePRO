from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from ImagePRO.human_analysis.face_analysis.face_detection import detect_faces
from ImagePRO.pre_processing import blur, grayscale, rotate, sharpen, resize
from ImagePRO.utils.image import Image

import random
import time

import cv2
import mediapipe as mp

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
    *,
    num_images: int = DEFAULT_NUM_IMAGES,
    start_index: int = DEFAULT_START_INDEX,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    apply_blur: bool = False,
    apply_grayscale: bool = False,
    apply_sharpen: bool = False,
    apply_rotate: bool = False,
    apply_resize: tuple = False,
    delay: float = DEFAULT_DELAY
) -> None:
    """Generate a dataset by capturing faces from webcam.

    Captures frames and saves preprocessed face images.
    Processing pipeline (if enabled):
    median blur → laplacian sharpen → grayscale → resize → random rotate

    Args:
        folder_path: Base directory for dataset.
        face_id: Subject identifier, creates folder "<base_dir>/<face_id>".
            Default: "unknown"
        num_images: Number of images to capture.
            Default: 200
        start_index: Starting number for filenames.
            Default: 0
        min_confidence: Face detection confidence threshold.
            Default: 0.7
        camera_index: OpenCV camera device index.
            Default: 0
        apply_blur: Apply median blur (size=3).
            Default: False
        apply_grayscale: Convert to single-channel.
            Default: False
        apply_sharpen: Apply Laplacian (coef=1.0).
            Default: False
        apply_rotate: Random rotation [-45°,45°].
            Default: False
        apply_resize: Optional (width,height).
            Default: False (no resize)
        delay: Time between captures (seconds).
            Default: 0.1

    Raises:
        TypeError: If input types are invalid
        ValueError: If numeric values are out of range
        FileExistsError: If output folder exists
        RuntimeError: If camera access fails
    """
    # Validate numeric parameters
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("'num_images' must be a positive integer")

    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("'start_index' must be a non-negative integer")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1")

    if not isinstance(delay, (int, float)) or delay < 0:
        raise ValueError("'delay' must be a non-negative number")

    # Setup output directory
    base_dir = Path(folder_path)
    face_folder = base_dir / str(face_id)

    try:
        face_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        raise FileExistsError(f"Output folder already exists: {face_folder}") from e

    # Initialize camera and face detector
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot access camera (index={camera_index})")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=False
    )

    saved_count = 0
    try:
        while saved_count < num_images:
            # Capture frame
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame, skipping...")
                continue

            # Apply preprocessing pipeline
            processed = frame

            if apply_blur:
                # Reduce noise while keeping facial features
                processed = blur.apply_average_blur(
                    image=Image.from_array(processed),
                    filter_size=3
                ).image

            if apply_sharpen:
                # Enhance edges with gentle sharpening
                processed = sharpen.apply_laplacian_sharpening(
                    image=Image.from_array(processed),
                    coefficient=1.0
                ).image

            if apply_grayscale:
                processed = grayscale.convert_to_grayscale(
                    image=Image.from_array(processed)
                ).image

            if apply_resize is not False:
                processed = resize.resize_image(
                    image=Image.from_array(processed),
                    new_size=apply_resize
                ).image

            if apply_rotate:
                # Apply random rotation with scaling
                angle = float(random.randint(-45, 45))
                scale = random.choice([1.0, 1.1, 1.2, 1.3])
                processed = rotate.rotate_image_custom(
                    image=Image.from_array(processed),
                    angle=angle,
                    scale=scale
                ).image

            # Save processed face image
            filename = f"{start_index + saved_count:04d}.jpg"
            output_path = face_folder / filename

            try:
                result = detect_faces(
                    image=Image.from_array(processed),
                    max_faces=1,
                    min_confidence=min_confidence,
                    face_mesh_obj=face_mesh
                )

                result.save_as_img(str(output_path))
                saved_count += 1

                if delay > 0:
                    time.sleep(delay)

            except ValueError:
                # Skip frames with no detected faces
                continue

    finally:
        # Clean up resources
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
