# Human Face Analysis

Face mesh, eye status, head pose, and face cropping with MediaPipe + OpenCV.  
Supports both single-image processing and real-time overlays.

## Features
- 468-point **Face Mesh** (full or subset draw)
- **Head pose** (yaw, pitch) via simple proportional landmark geometry
- **Eye status** (open/closed) via EAR on right eye
- **Face cropping** from facial outline landmarks
- Optional simultaneous annotated-image saving and CSV export

## I/O Conventions
- Provide either `src_image_path` or `src_np_image` (BGR). If both set, `src_np_image` is used.
- Saving:
  - `output_image_path` for annotated images
  - `output_csv_path` for landmark CSV
  - Functions print save logs and still return in-memory results.
- Live helpers open webcam; press **ESC** to exit.

## Submodules & Functions
### `face_mesh_analysis.py`
- `analyze_face_mesh(max_faces=1, min_confidence=0.7, landmarks_idx=None, src_image_path=None, src_np_image=None, output_image_path=None, output_csv_path=None, face_mesh_obj=None) -> tuple[np.ndarray, list]`
- `analyze_face_mesh_live(max_faces=1, min_confidence=0.7) -> None`

### `head_pose_estimation.py`
- `estimate_head_pose(max_faces=1, min_confidence=0.7, src_image_path=None, src_np_image=None, output_csv_path=None, face_mesh_obj=None) -> list[list[float]] | str`
- `estimate_head_pose_live(max_faces=1, min_confidence=0.7) -> None`

### `eye_status.py`
- `analyze_eye_status(min_confidence=0.7, src_image_path=None, src_np_image=None, face_mesh_obj=None, threshold=0.2) -> bool`
- `analyze_eye_status_live(min_confidence=0.7, threshold=0.2) -> None`

### `face_crop.py`
- `detect_faces(max_faces=1, min_confidence=0.7, src_image_path=None, src_np_image=None, output_image_path=None, face_mesh_obj=None) -> list[np.ndarray] | str`

## Quick Start
```python
from human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh
from human_analysis.face_analysis.head_pose_estimation import estimate_head_pose
from human_analysis.face_analysis.eye_status import analyze_eye_status
from human_analysis.face_analysis.face_crop import detect_faces

# Mesh + CSV
annotated, landmarks = analyze_face_mesh(
    src_image_path="face.jpg",
    output_image_path="mesh.jpg",
    output_csv_path="landmarks.csv"
)

# Head pose (list of [face_id, yaw, pitch])
pose = estimate_head_pose(src_np_image=annotated, output_csv_path="pose.csv")

# Eye status (right eye)
is_open = analyze_eye_status(src_np_image=annotated, threshold=0.22)

# Crop faces from outline
crops = detect_faces(src_np_image=annotated, output_image_path="crop.jpg")
```

## Error Handling
- `ValueError` / `TypeError`: invalid inputs or missing landmarks
- `RuntimeError`: webcam unavailable (live)
- From `IOHandler`:
  - `FileNotFoundError`, `TypeError`, `ValueError` on load
  - `IOError` on save

## Notes
- MediaPipe face coordinates are **normalized** `[0,1]`. Convert to pixels when needed.
- Head pose here is a **lightweight heuristic** (not PnP). For metric pose, integrate a 3D model + `cv2.solvePnP`.
- Reuse `face_mesh_obj` across calls to reduce init overhead in batch workflows.
