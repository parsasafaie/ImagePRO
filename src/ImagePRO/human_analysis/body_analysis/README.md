# Human Body Analysis

Pose and hand landmark detection for images and live webcam input (MediaPipe).  
Designed for lightweight overlays, CSV export, and downstream tasks like gesture analysis.

## Features
- 33-point **Pose** landmarks + connections
- 21-point **Hands** landmarks per hand (up to 2 hands)
- Optional **simultaneous** annotated-image saving and CSV export
- Live utilities for quick demos and prototyping

## I/O Conventions
- For image functions, provide:
  - `src_image_path: str` or `src_np_image: np.ndarray` (BGR). `src_np_image` wins if both set.
- Saving:
  - `output_image_path`: saves annotated image (prints a log, returns image)
  - `output_csv_path`: saves flattened landmarks CSV (prints a log)
- Live functions open the default webcam; press **ESC** to exit.

## Submodules & Functions
### `body_pose_estimation.py`
- `detect_body_pose(model_accuracy=0.7, landmarks_idx=None, src_image_path=None, src_np_image=None, output_image_path=None, output_csv_path=None, pose_obj=None) -> tuple[np.ndarray, list]`
- `detect_body_pose_live() -> None`

### `hand_tracking.py`
- `detect_hands(max_hands=2, min_confidence=0.7, landmarks_idx=None, src_image_path=None, src_np_image=None, output_image_path=None, output_csv_path=None, hands_obj=None) -> tuple[np.ndarray, list]`
- `detect_hands_live(max_hands=2, min_confidence=0.7) -> None`

## Quick Start
```python
from human_analysis.body_analysis.body_pose_estimation import detect_body_pose
from human_analysis.body_analysis.hand_tracking import detect_hands

# Single image
img_annotated, pose_landmarks = detect_body_pose(
    src_image_path="person.jpg",
    output_image_path="pose.jpg",
    output_csv_path="pose.csv"
)

# Hands
hand_img, hands_landmarks = detect_hands(
    src_np_image=img_annotated,
    output_image_path="hands.jpg",
    output_csv_path="hands.csv"
)

# Live
from human_analysis.body_analysis.body_pose_estimation import detect_body_pose_live
detect_body_pose_live()
```

## Error Handling
- `ValueError` / `TypeError`: invalid arguments (e.g., counts, confidences, indices)
- `RuntimeError`: webcam unavailable (live)
- From `IOHandler`:
  - `FileNotFoundError`, `TypeError`, `ValueError` on load
  - `IOError` on save

## Notes
- Coordinates from MediaPipe are **normalized** in `[0,1]` (x,y,z).  
  Multiply by image `width/height` for pixel coordinates when needed.
- For live, models are created with `static_image_mode=False` for better throughput.
