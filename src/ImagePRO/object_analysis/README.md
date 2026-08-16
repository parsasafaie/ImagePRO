# Object Analysis Module

YOLO-based object detection with multiple accuracy levels and flexible model support.

## Features

- **Multiple YOLO Models**: Nano to extra-large accuracy levels
- **Flexible Model Loading**: Pre-trained or custom model support
- **Rich Output**: Annotated images and detailed detection data
- **Performance Optimized**: GPU acceleration when available

## I/O Conventions

- **Input**: A `Image` instance created by path or array
- **Output**: A `Result` instance contains image(np.ndarray), data(any other data like detections) and meta(some additional info about process)
- **Model Management**: Automatic model loading or custom model injection

## Available Functions

### **Core Detection**
- **`object_detection.py`**: YOLO-based object detection with multiple accuracy levels

## Quick Start

```python
from ImagePRO.object_analysis.object_detection import detect_objects
from ImagePRO.utils.image import Image
import cv2

# Load image
image = Image.from_path("image.jpg")  # Or -> image = Image.from_array(np_image)

# Detect objects with nano model (fastest)
result = detect_objects(
    image=image,
    accuracy_level=1  # 1=nano, 2=small, 3=medium, 4=large, 5=extra-large
)

print(type(result))  # Should print: <class 'ImagePRO.utils.result.Result'>
print(result.data)   # Should print: List of detections
print(result.meta)   # Should print: Operation metadata

# Display result
cv2.imshow("Detections", result.image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use custom model
from ultralytics import YOLO
custom_model = YOLO("path/to/custom.pt")

result = detect_objects(
    image=image,
    model=custom_model
)
```

## Output Formats

### **Detection Results**
- **Return Value**: `Result` with annotated image and detection list
- **Data Format**: `[class_id, [x1, y1, x2, y2], confidence]` per detection
- **Coordinates**: Bounding box coordinates normalized to [0, 1] (multiply by image width/height for pixels)

### **Accuracy Levels**
- **Level 1**: `yolo11n.pt` - Fastest, lowest accuracy
- **Level 2**: `yolo11s.pt` - Small, balanced
- **Level 3**: `yolo11m.pt` - Medium, good balance
- **Level 4**: `yolo11l.pt` - Large, high accuracy
- **Level 5**: `yolo11x.pt` - Extra-large, highest accuracy

## Error Handling

- **`ValueError`**: Invalid accuracy level or parameters
- **`TypeError`**: Incorrect input types
- **`FileNotFoundError`**: Image file not found
- **`RuntimeError`**: Model loading failures

## Technical Notes

- **YOLO Integration**: Uses Ultralytics YOLO implementation (YOLO11)
- **Model Loading**: A new model instance is created per call unless one is passed via `model=`; reuse a pre-loaded model in loops for speed
- **GPU Support**: Ultralytics uses CUDA automatically when available
- **Custom Models**: Support for custom-trained YOLO models

## Related Modules

- See [Utils Module](../utils/README.md) for `Image` and `Result` class documentation
- See [Pre-processing Module](../pre_processing/README.md) for image manipulation
- See [Human Analysis Module](../human_analysis/README.md) for face/body analysis
