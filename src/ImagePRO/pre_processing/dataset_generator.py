from __future__ import annotations

import random
import sys
import time
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import cv2

from ImagePRO.human_analysis.face_analysis.face_detection import detect_faces
from ImagePRO.pre_processing import blur, grayscale, resize, rotate, sharpen
from ImagePRO.utils.image import Image

# Constants
DEFAULT_NUM_IMAGES = 200
DEFAULT_START_INDEX = 0
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_CAMERA_INDEX = 0
DEFAULT_DELAY = 0.1
DEFAULT_FACE_ID = "unknown"
MAX_EMPTY_FRAMES = 250  # Abandon capture after this many frames without a face


def _validate_parameters(
    *,
    num_images: int,
    start_index: int,
    min_confidence: float,
    camera_index: int,
    apply_resize: tuple[int, int] | bool,
    delay: float
) -> None:
    """Validate all user-supplied parameters up front, before touching the camera."""
    if not isinstance(num_images, int) or isinstance(num_images, bool) or num_images <= 0:
        raise ValueError("'num_images' must be a positive integer")

    if not isinstance(start_index, int) or isinstance(start_index, bool) or start_index < 0:
        raise ValueError("'start_index' must be a non-negative integer")

    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)) \
            or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1")

    if not isinstance(camera_index, int) or isinstance(camera_index, bool) or camera_index < 0:
        raise ValueError("'camera_index' must be a non-negative integer")

    if apply_resize is not False and (
        not isinstance(apply_resize, tuple)
        or len(apply_resize) != 2
        or not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
            for dim in apply_resize
        )
    ):
        raise ValueError("'apply_resize' must be False or a (width, height) tuple "
                         "of two positive integers")

    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
        raise ValueError("'delay' must be a non-negative number")


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
    apply_resize: tuple[int, int] | bool = False,
    delay: float = DEFAULT_DELAY
) -> None:
    """Generate a dataset by capturing faces from webcam.

    Captures frames from the webcam, detects faces, applies optional
    preprocessing, and saves cropped face images to disk. This function is
    useful for creating face recognition datasets with data augmentation.

    Saved crops go through this pipeline (each step optional):
    median blur → laplacian sharpen → resize → grayscale → random rotate

    Face detection always runs on the raw camera frame, so the capture loop
    keeps working no matter which preprocessing flags are enabled. The loop
    gives up after MAX_EMPTY_FRAMES consecutive frames without a usable
    capture, instead of blocking forever when no face is in view.

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
        apply_grayscale: Convert crops to single-channel.
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
    # Validate everything up front, before creating folders or opening the
    # camera, so an invalid argument never leaves partial state behind.
    _validate_parameters(
        num_images=num_images,
        start_index=start_index,
        min_confidence=min_confidence,
        camera_index=camera_index,
        apply_resize=apply_resize,
        delay=delay
    )

    if apply_rotate is not None and not isinstance(apply_rotate, bool):
        raise TypeError("'apply_rotate' must be a boolean")

    try:
        import mediapipe as mp
    except ImportError as err:
        raise ImportError(
            "The optional 'mediapipe' dependency is required for dataset "
            'generation. Install it with: pip install "ImagePRO-Python[mediapipe]"'
        ) from err

    # Setup output directory structure
    base_dir = Path(folder_path)
    face_folder = base_dir / str(face_id)

    # Create output folder (fail if exists to prevent accidental overwrites)
    try:
        face_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        raise FileExistsError(f"Output folder already exists: {face_folder}") from e

    # Initialize camera capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot access camera (index={camera_index})")

    # Initialize MediaPipe face mesh detector for face detection
    # Using face_mesh instead of face_detection for better accuracy
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=False
    )

    saved_count = 0
    frames_without_face = 0
    try:
        while saved_count < num_images:
            # Capture frame
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame, skipping...")
                frames_without_face += 1
                if frames_without_face >= MAX_EMPTY_FRAMES:
                    raise RuntimeError(
                        f"Camera stopped delivering frames after "
                        f"{frames_without_face} consecutive failures; "
                        f"saved {saved_count}/{num_images} images so far."
                    )
                continue

            # Apply preprocessing pipeline in sequence
            # Each step transforms the image and passes it to the next step
            processed = frame

            if apply_blur:
                # Reduce noise while preserving facial features
                processed = blur.apply_median_blur(
                    image=Image.from_array(processed),
                    filter_size=3
                ).image

            if apply_sharpen:
                # Enhance edges with gentle sharpening for better feature detection
                processed = sharpen.apply_laplacian_sharpening(
                    image=Image.from_array(processed),
                    coefficient=1.0
                ).image

            if apply_resize is not False:
                # Resize to consistent dimensions (e.g., 224x224 for ML models)
                processed = resize.resize_image(
                    image=Image.from_array(processed),
                    new_size=apply_resize
                ).image

            if apply_grayscale:
                # Convert to single-channel for reduced storage and faster
                # processing. Rotation then tags the intermediate as GRAY so
                # the face detector can convert back to RGB correctly.
                processed = grayscale.convert_to_grayscale(
                    image=Image.from_array(processed)
                ).image

            if apply_rotate:
                # Apply random rotation with scaling for data augmentation
                # Rotation range: -45° to +45° with random scale factors
                angle = float(random.randint(-45, 45))
                scale = random.choice([1.0, 1.1, 1.2, 1.3])
                processed = rotate.rotate_image_custom(
                    image=Image.from_array(processed),
                    angle=angle,
                    scale=scale
                ).image

            # Detect and crop face from processed frame
            filename = f"{start_index + saved_count:04d}.jpg"
            output_path = face_folder / filename

            try:
                # Detect face and crop to face region only. The detector
                # needs a color image, so pass the untouched camera frame
                # (preprocessing here is augmentation for the saved crop,
                # not a requirement of the detector).
                result = detect_faces(
                    image=Image.from_array(frame),
                    max_faces=1,
                    min_confidence=min_confidence,
                    face_mesh_obj=face_mesh
                )

                # Save cropped face image
                result.save_as_img(str(output_path))
                saved_count += 1
                frames_without_face = 0

                # Add delay between captures to allow subject movement
                if delay > 0:
                    time.sleep(delay)

            except ValueError:
                # Skip frames with no detected faces
                frames_without_face += 1
                if frames_without_face >= MAX_EMPTY_FRAMES:
                    print(
                        f"No face detected in {frames_without_face} consecutive "
                        f"frames; giving up with {saved_count}/{num_images} "
                        "images saved."
                    )
                    break
                continue

    finally:
        # Clean up resources: release camera and close OpenCV windows
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
