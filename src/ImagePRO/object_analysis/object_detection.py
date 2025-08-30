from ultralytics import YOLO
import sys
from pathlib import Path

# Add parent directory to path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

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
    *,
    image: Image | None = None,
    model=None,
    accuracy_level: int = DEFAULT_ACCURACY_LEVEL,
    show_result: bool = False
):
    """
    Run object detection on a single image using Ultralytics YOLO.

    Parameters
    ----------
    image : Image
        Image instance (BGR) to process.
    model : ultralytics.engine.model.YOLO | None, optional
        A pre-loaded YOLO model instance. If provided, `accuracy_level` is ignored.
        If None, a model is created based on `accuracy_level`.
    accuracy_level : int, default=1
        Model size preset in {1..5} mapping to:
        1 -> "yolo11n.pt", 2 -> "yolo11s.pt", 3 -> "yolo11m.pt",
        4 -> "yolo11l.pt", 5 -> "yolo11x.pt".
    show_result : bool, default=False
        If True, shows the result window via `result.show()` (may require a GUI environment).

    Returns
    -------
    Result
        `data` is a list of [box_class, [x1, y1, x2, y2], conf] per object; `image` is the original image with bounding boxes drawn around detected objects.

    Raises
    ------
    ValueError
        If `accuracy_level` is not in {1..5}, or image is invalid
    """
    if not isinstance(image, Image):
        raise ValueError("'image' must be an instance of Image.")

    # Create a model from preset if not provided by caller
    if model is None:
        if accuracy_level not in MODEL_MAPPING:
            raise ValueError(f"'accuracy_level' must be in {list(MODEL_MAPPING.keys())}, got {accuracy_level}")
        
        model_name = MODEL_MAPPING[accuracy_level]
        model = YOLO(model=model_name)

    # Load image from path or ndarray using the shared IO helper
    frame = image._data

    # Run inference (first result only)
    result = model(frame)[0]
    boxes = result.boxes

    # Collect rows as: [class_id, [x1n, y1n, x2n, y2n], confidence]
    lines = []
    for box in boxes:
        box_class = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = [float(c) for c in box.xyxyn.squeeze().tolist()]
        lines.append([box_class, [x1, y1, x2, y2], conf])

    # Optional visualization and saving
    if show_result:
        result.show()
    
    # Return the Result instance
    return Result(image=result.plot(), data=lines, meta={"source"})
