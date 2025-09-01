# ImagePRO

> **Professional & Modular Image Processing Library in Python**

**ImagePRO** is a clean, modular, and easy-to-use Python library for image processing tasks, built with OpenCV, MediaPipe, YOLO and designed to be extensible for developers.

Whether you're working on computer vision pipelines, preprocessing images for AI models, or simply automating batch image edits — **ImagePRO** gives you powerful tools with minimal effort.

## ✨ Features

### **Image I/O & Management**
- Flexible input/output handling (file paths or numpy arrays)
- Batch processing capabilities
- Multiple format support (JPEG, PNG, CSV, etc.)

### **Pre-processing & Enhancement**
- **Basic Operations**: Resize, crop, rotate (90°, 180°, 270°, custom angles), grayscale conversion
- **Filtering**: Blur filters (average, Gaussian, median, bilateral), sharpening filters (Laplacian, Unsharp Masking)
- **Enhancement**: Contrast enhancement (CLAHE, GHE, stretching)
- **Dataset Generation**: Automated image capture with preprocessing pipeline

### **Human Analysis**
- **Face Analysis**: 468-point mesh, head pose estimation, eye status detection, face comparison, face cropping
- **Body Analysis**: Pose estimation, hand tracking (21 landmarks)
- **Real-time Processing**: Live webcam analysis for all modules

### **Object Detection**
- **YOLO Integration**: Multiple accuracy levels (nano to extra-large)
- **Flexible Models**: Pre-trained or custom model support
- **Batch Processing**: Efficient handling of multiple images

## 🚀 Installation

### From PyPI
```bash
pip install ImagePRO-Python
```

### From Source
```bash
git clone https://github.com/parsasafaie/ImagePRO.git
cd ImagePRO

python -m venv .venv

source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

# Base dependencies
pip install -r requirements/base.txt

# Optional dependencies
# For YOLO object detection
pip install -r requirements/yolo.txt

# For MediaPipe human analysis
pip install -r requirements/mediapipe.txt

# For InsightFace advanced face analysis
pip install -r requirements/insightface.txt

# Or install everything
pip install -r requirements/full.txt
```

See the [Directory Structure](/PROJECT_STRUCTURE.md#directory-structure) section in [PROJECT_STRUCTURE.md](/PROJECT_STRUCTURE.md) for details on which modules need which requirements.

## 📖 Quick Start

```python
from ImagePRO.pre_processing.blur import apply_average_blur
from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh
from ImagePRO.human_analysis.body_analysis.body_pose_estimation import detect_body_pose
from ImagePRO.object_analysis.object_detection import detect_objects

from ImagePRO.utils.image import Image

image = Image.from_path("person_and_objects.jpg")

# Apply Average blur
blur_result = apply_average_blur(
    image=image, 
)

# Analyze face mesh
face_mesh_result = analyze_face_mesh(
    image=image
)

# Detect body pose
body_pose_result = detect_body_pose(
    image=image
)   

# Detect objects                
object_detection_result = detect_objects(
    image=image,
)

# There is 4 example from each module, there is more functions and utilities in each module, and more customization options in each function. explore the codebase for more details.
# If you have any feature requests or suggestions, please open an issue on the GitHub repository.
```

## 📚 Documentation

Each module includes comprehensive documentation:
- **Pre-processing**: Image manipulation and enhancement
- **Human Analysis**: Face and body analysis tools
- **Object Analysis**: YOLO-based detection
- **Utils**: Shared utilities and I/O handling

## 🏗️ Architecture

ImagePRO is built with a modular architecture:
- **Clean separation of concerns**
- **Consistent API patterns** across all modules
- **Shared utilities** for common operations
- **Professional error handling** and validation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License 
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
