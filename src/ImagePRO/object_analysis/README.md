# Object Analysis

Tools for detecting objects in images using YOLO models.

## Features
- Accepts image from `src_image_path` or `src_np_image` (BGR)
- Optional `output_image_path` to save annotated image
- Optional `output_csv_path` to save detections
- Can use pre-loaded YOLO model or load by `accuracy_level` (1–5)

## Function
### `object_detection.py`
- `detect_objects(model=None, accuracy_level=1, src_image_path=None, src_np_image=None, output_image_path=None, output_csv_path=None, show_result=False) -> tuple[np.ndarray, list[list]]`  
  Runs YOLO detection and returns `(annotated_image, detections)`.  
  Each detection: `[class_id, [x1n, y1n, x2n, y2n], confidence]` with normalized coords.

## Example
```python
from object_analysis.object_detection import detect_objects
annotated, dets = detect_objects(accuracy_level=3, src_image_path="img.jpg",
                                 output_image_path="out.jpg", output_csv_path="out.csv")
