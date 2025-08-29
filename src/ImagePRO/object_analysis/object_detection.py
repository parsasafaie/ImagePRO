from ultralytics import YOLO
import sys
from pathlib import Path

# Add parent directory to path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler

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
    model=None,
    accuracy_level: int = DEFAULT_ACCURACY_LEVEL,
    src_image_path: str | None = None,
    src_np_image=None,
    output_image_path: str | None = None,
    output_csv_path: str | None = None,
    show_result: bool = False
):
    """
    Run object detection on a single image using Ultralytics YOLO.

    Parameters
    ----------
    model : ultralytics.engine.model.YOLO | None, optional
        A pre-loaded YOLO model instance. If provided, `accuracy_level` is ignored.
        If None, a model is created based on `accuracy_level`.
    accuracy_level : int, default=1
        Model size preset in {1..5} mapping to:
        1 -> "yolo11n.pt", 2 -> "yolo11s.pt", 3 -> "yolo11m.pt",
        4 -> "yolo11l.pt", 5 -> "yolo11x.pt".
    src_image_path : str | None, optional
        Path to the input image on disk. Used by IOHandler if `src_np_image` is None.
    src_np_image : np.ndarray | None, optional
        Image array (BGR). If provided, it is preferred over `src_image_path`.
    output_image_path : str | None, optional
        If provided, saves the annotated image (result.plot()) to this path.
    output_csv_path : str | None, optional
        If provided, saves detections as rows of `[class_id, [x1n, y1n, x2n, y2n], confidence]`.
        Note: coordinates are normalized in [0, 1].
    show_result : bool, default=False
        If True, shows the result window via `result.show()` (may require a GUI environment).

    Returns
    -------
    tuple
        (annotated_image: np.ndarray, lines: list[list])
        where `annotated_image` is the result of `result.plot()` and
        `lines` is a list of detection rows saved to CSV when requested.

    Raises
    ------
    ValueError
        If `accuracy_level` is not in {1..5}, or if image loading fails inside IOHandler.
    """
    # Create a model from preset if not provided by caller
    if model is None:
        if accuracy_level not in MODEL_MAPPING:
            raise ValueError(f"'accuracy_level' must be in {list(MODEL_MAPPING.keys())}, got {accuracy_level}")
        
        model_name = MODEL_MAPPING[accuracy_level]
        model = YOLO(model=model_name)

    # Load image from path or ndarray using the shared IO helper
    frame = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)

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
    if output_image_path:
        print(IOHandler.save_image(np_image=result.plot(), result_path=output_image_path))
    if output_csv_path:
        print(IOHandler.save_csv(data=lines, result_path=output_csv_path))

    # Return the plotted image and the CSV-friendly rows
    return result.plot(), lines
