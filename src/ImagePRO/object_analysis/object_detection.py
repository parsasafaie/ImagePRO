from ultralytics import YOLO
import sys
from pathlib import Path
import cv2

# Add parent directory to path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler

def detect_objects(model=None, accuracy_level=1, src_image_path=None, src_np_image=None, output_image_path=None, output_csv_path=None, show_result=False):
    if model is None:
        if accuracy_level==1:
            model_name = "yolo11n.pt"
        elif accuracy_level==2:
            model_name = "yolo11s.pt"
        elif accuracy_level==3:
            model_name = "yolo11m.pt"
        elif accuracy_level==4:
            model_name = "yolo11l.pt"
        elif accuracy_level==5:
            model_name = "yolo11x.pt"
        else:
            raise ValueError('Unknown accuracy level.')

        model = YOLO(model=model_name)

    frame = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    result = model(frame)[0]

    if show_result:
        result.show()
    if output_image_path:
        IOHandler.save_image(np_image=result.plot(), result_path=output_image_path)
    if output_csv_path:
        boxes = result.boxes
        lines = []
        for box in boxes:
            box_class = int(box.cls)
            conf  = float(box.conf)
            x1, y1, x2, y2 = [float(c) for c in box.xyxyn.squeeze().tolist()]
            lines.append([box_class,[x1, y1, x2, y2], conf])
        IOHandler.save_csv(data=lines, result_path=output_csv_path)

    return result