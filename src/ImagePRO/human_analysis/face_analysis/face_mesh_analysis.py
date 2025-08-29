import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Add parent directory to path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
DEFAULT_MAX_FACES = 1
DEFAULT_MIN_CONFIDENCE = 0.7
TOTAL_FACE_LANDMARKS = 468


def analyze_face_mesh(
    max_faces: int = DEFAULT_MAX_FACES,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    landmarks_idx: list | None = None,
    image: Image | None = None,
    face_mesh_obj=None,
):
    """
    Detect facial landmarks using MediaPipe FaceMesh on a single image (path or ndarray).

    Parameters
    ----------
    max_faces : int, default=1
        Maximum number of faces to detect.
    min_confidence : float, default=0.7
        Minimum detection confidence in [0, 1].
    landmarks_idx : list[int] | None, optional
        Specific landmark indices to extract/draw. If ``None``, uses the full
        468-point mesh.
    src_image_path : str | None, optional
        Path to the input image (BGR/RGB supported by ``IOHandler.load_image``).
    src_np_image : np.ndarray | None, optional
        Image array in BGR (as returned by OpenCV). If both ``src_image_path`` and
        ``src_np_image`` are provided, the ndarray takes precedence.
    output_image_path : str | None, optional
        If provided (e.g. "out.jpg"/"out.png"), the annotated image is saved there.
    output_csv_path : str | None, optional
        If provided (e.g. "landmarks.csv"), landmark coordinates are saved there as
        rows: ``[face_id, index, x, y, z]`` with normalized x/y/z from MediaPipe.
    face_mesh_obj : mediapipe.python.solutions.face_mesh.FaceMesh | None, optional
        Reusable external FaceMesh instance. If ``None``, a new one is created
        with ``static_image_mode=True`` and ``refine_landmarks=True``.

    Returns
    -------
    tuple
        ``(annotated_image: np.ndarray, all_landmarks: list[list[list[float]]])``
        where ``all_landmarks`` is a list per face, each containing rows of
        ``[face_id, index, x, y, z]``.

    Raises
    ------
    ValueError
        * If inputs are invalid or no face landmarks are detected (this function).
        * From ``IOHandler.load_image`` when both inputs are ``None`` or image loading fails.
    TypeError
        * If ``landmarks_idx`` is not a list of integers (this function).
        * From ``IOHandler.load_image``/``save_image``/``save_csv`` on invalid argument types.
    FileNotFoundError
        From ``IOHandler.load_image`` when ``image_path`` does not exist.
    IOError
        From ``IOHandler.save_image`` when saving the image fails.

    Notes
    -----
    * The x/y/z coordinates from MediaPipe are **normalized** to [0, 1] in image
      space; multiply by width/height to get pixel coordinates.
    * When ``landmarks_idx`` covers all 468 points, the full tessellation is drawn.
      Otherwise, only the requested points are highlighted.
    """

    # --- Validate inputs -----------------------------------------------------
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be a float between 0 and 1.")

    if landmarks_idx is not None:
        if not isinstance(landmarks_idx, list) or not all(isinstance(i, int) for i in landmarks_idx):
            raise TypeError("'landmarks_idx' must be a list of integers or None.")

    if image is None or not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")
    # --- Create or reuse FaceMesh instance ----------------------------------
    if face_mesh_obj is None:
        face_Mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=min_confidence,
            refine_landmarks=True,
            static_image_mode=True,
        )
    else:
        face_Mesh = face_mesh_obj

    # --- Load image (path or ndarray) ---------------------------------------
    np_image = image._data

    # Convert BGR -> RGB for MediaPipe
    annotated_image = np_image.copy()
    rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    results = face_Mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        raise ValueError("No face landmarks detected.")

    landmarks_idx = landmarks_idx or list(range(468))

    all_landmarks = []

    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
        # Draw either the full tessellation or specific landmark dots
        if len(landmarks_idx) == TOTAL_FACE_LANDMARKS:
            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
        else:
            h, w = annotated_image.shape[:2]
            for idx in landmarks_idx:
                lm = face_landmarks.landmark[idx]
                cx, cy = int(w * lm.x), int(h * lm.y)
                cv2.circle(annotated_image, (cx, cy), 3, (0, 0, 255), -1)

        # Collect normalized coordinates for requested indices
        face_data = [
            [face_id, idx, face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y, face_landmarks.landmark[idx].z]
            for idx in landmarks_idx
        ]
        all_landmarks.append(face_data)

    # --- Output handling -----------------------------------------------------

    return Result(image=annotated_image, data=all_landmarks, meta={"source":image, "operation":"analyze_face_mesh", "landmarks_idx": landmarks_idx, "max_faces": max_faces, "min_confidence": min_confidence})


def analyze_face_mesh_live(max_faces: int = DEFAULT_MAX_FACES, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
    """
    Launch webcam preview with real-time FaceMesh overlay.

    Parameters
    ----------
    max_faces : int, default=1
        Max number of faces to detect per frame.
    min_confidence : float, default=0.7
        Minimum detection confidence in [0, 1].

    Raises
    ------
    ValueError
        If arguments are invalid.
    RuntimeError
        If the webcam cannot be opened.

    Notes
    -----
    * Uses ``static_image_mode=False`` for better performance on video streams.
    * Press **ESC** to exit the preview window.
    """
    if not isinstance(max_faces, int) or max_faces <= 0:
        raise ValueError("'max_faces' must be a positive integer.")

    if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
        raise ValueError("'min_confidence' must be between 0 and 1.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    # For live video, static_image_mode=False is preferred (tracking across frames)
    face_Mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence,
        refine_landmarks=True,
        static_image_mode=False,
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Skipping empty frame.")
                continue

            try:
                result_image, _ = analyze_face_mesh(
                    max_faces=max_faces,
                    min_confidence=min_confidence,
                    src_np_image=frame,
                    face_mesh_obj=face_Mesh,
                )
            except ValueError:
                # No face detected in this frame; show raw frame
                result_image = frame

            cv2.imshow("ImagePRO - Face Mesh", result_image)

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_face_mesh_live(max_faces=1)
