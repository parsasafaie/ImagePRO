from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from ultralytics import YOLO

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
DEFAULT_ACCURACY_LEVEL = 1
DEFAULT_CONFIDENCE = 0.5
MODEL_MAPPING = {
    1: "yolo11n.pt",
    2: "yolo11s.pt", 
    3: "yolo11m.pt",
    4: "yolo11l.pt",
    5: "yolo11x.pt"
}


def detect_objects(
    image: Image,
    *,
    model: YOLO | None = None,
    accuracy_level: int = DEFAULT_ACCURACY_LEVEL,
    show_result: bool = False
) -> Result:
    """Detect objects in an image using YOLO models.

    Uses Ultralytics YOLO for object detection with customizable
    model size and accuracy tradeoffs.

    Args:
        image: Input image to process.
        model: Pre-loaded YOLO model. If None, loads based on accuracy_level.
            Default: None
        accuracy_level: Model size/accuracy preset (1-5):
            1: yolo11n (fastest)
            2: yolo11s
            3: yolo11m
            4: yolo11l
            5: yolo11x (most accurate)
            Default: 1
        show_result: Show detection visualization window.
            Default: False

    Returns:
        Result object with detections and metadata:
        - image: Original with bounding boxes drawn
        - data: List of [class_id, [x1,y1,x2,y2], confidence]
        - meta: Operation info and model used

    Raises:
        TypeError: If image is not an Image instance
        ValueError: If accuracy_level not in range 1-5
    """
    if not isinstance(image, Image):
        raise TypeError("'image' must be an Image instance")

    # Initialize model
    model_name = None
    if model is None:
        if accuracy_level not in MODEL_MAPPING:
            raise ValueError(
                f"'accuracy_level' must be in {list(MODEL_MAPPING.keys())}, "
                f"got {accuracy_level}"
            )
        
        model_name = MODEL_MAPPING[accuracy_level]
        model = YOLO(model=model_name)

    # Run inference
    result = model(image._data.copy())[0]
    boxes = result.boxes

    # Process detections
    detections = []
    for box in boxes:
        box_class = int(box.cls)
        confidence = float(box.conf)
        x1, y1, x2, y2 = [float(c) for c in box.xyxyn.squeeze().tolist()]
        detections.append([box_class, [x1, y1, x2, y2], confidence])

    # Show visualization if requested
    if show_result:
        result.show()
    
    return Result(
        image=result.plot(),
        data=detections,
        meta={
            "source": image,
            "operation": "detect_objects",
            "model": model_name or "custom"
        }
    )
