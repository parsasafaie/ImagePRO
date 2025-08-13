from ultralytics import YOLO
import sys
from pathlib import Path

# Add parent directory to path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler

def detect_objects(src_image_path=None, src_np_image=None, output_image_path=None, show_result=False):
    model = YOLO("yolo11n.pt") 
    frame = IOHandler.load_image(image_path=src_image_path, np_image=src_np_image)
    result = model(frame)[0]

    if show_result:
        result.show()
    if output_image_path:
        IOHandler.save_image(np_image=result.plot(), result_path=output_image_path)
    
    return result
