# Object Analysis Module

YOLO-based object detection with multiple accuracy levels and flexible model support.

## ✨ Features

- **Multiple YOLO Models**: Nano to extra-large accuracy levels
- **Flexible Model Loading**: Pre-trained or custom model support
- **Batch Processing**: Efficient handling of multiple images
- **Rich Output**: Annotated images and detailed detection data
- **Performance Optimized**: GPU acceleration when available

## 🔧 I/O Conventions

- **Input**: Support for both file paths and numpy arrays
- **Output**: Optional saving of annotated images and detection data
- **Precedence**: Numpy arrays take priority when both inputs provided
- **Model Management**: Automatic model loading or custom model injection

## 📚 Available Functions

### **Core Detection**
- **`object_detection.py`**: YOLO-based object detection with multiple accuracy levels

## 🚀 Quick Start

```python
from ImagePRO.object_analysis.object_detection import detect_objects

# Detect objects with nano model (fastest)
results = detect_objects(
    src_image_path="image.jpg",
    accuracy_level=1,  # 1=nano, 2=small, 3=medium, 4=large, 5=extra-large
    confidence=0.5,
    output_image_path="detected.jpg",
    output_csv_path="detections.csv"
)

# Use custom model
from ultralytics import YOLO
custom_model = YOLO("path/to/custom.pt")

results = detect_objects(
    src_np_image=image_array,
    model=custom_model,
    output_image_path="custom_detected.jpg"
)
```

## 📊 Output Formats

### **Detection Results**
- **Return Value**: YOLO Results object with detection data
- **CSV Format**: `[image_id, class_id, class_name, confidence, x1, y1, x2, y2]`
- **Coordinates**: Bounding box coordinates in pixels

### **Accuracy Levels**
- **Level 1**: `yolo11n.pt` - Fastest, lowest accuracy
- **Level 2**: `yolo11s.pt` - Small, balanced
- **Level 3**: `yolo11m.pt` - Medium, good balance
- **Level 4**: `yolo11l.pt` - Large, high accuracy
- **Level 5**: `yolo11x.pt` - Extra-large, highest accuracy

## ⚠️ Error Handling

- **`ValueError`**: Invalid accuracy level or parameters
- **`TypeError`**: Incorrect input types
- **`FileNotFoundError`**: Image file not found
- **`RuntimeError`**: Model loading failures

## 📝 Technical Notes

- **YOLO Integration**: Uses Ultralytics YOLO implementation
- **Model Caching**: Models are cached after first load
- **GPU Support**: Automatic CUDA detection and utilization
- **Performance**: Optimized for both CPU and GPU inference
