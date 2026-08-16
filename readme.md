# ImagePRO

> **Professional & Modular Image Processing Library in Python**

**ImagePRO** is a clean, modular, and easy-to-use Python library for image processing tasks, built with OpenCV, MediaPipe, YOLO, and InsightFace. Designed to be extensible for developers, ImagePRO provides a consistent API across all modules with comprehensive error handling and professional-grade documentation.

Whether you're working on computer vision pipelines, preprocessing images for AI models, or simply automating batch image edits — **ImagePRO** gives you powerful tools with minimal effort.

## Features

### **Image I/O & Management**
- Flexible input/output handling (file paths or numpy arrays)
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

## Installation

### From PyPI
```bash
pip install ImagePRO-Python

# Or with optional extras:
pip install "ImagePRO-Python[yolo]"         # YOLO object detection
pip install "ImagePRO-Python[mediapipe]"    # MediaPipe human analysis
pip install "ImagePRO-Python[insightface]"  # InsightFace face comparison
pip install "ImagePRO-Python[full]"         # Everything
pip install "ImagePRO-Python[dev]"          # Test tooling (pytest)
```

> **Note**: The base install is lightweight — AI backends (MediaPipe, Ultralytics, InsightFace) are optional and imported lazily, only when you call a function that actually needs them. `import ImagePRO` never loads these dependencies. If a backend is missing, the function you call raises an `ImportError` telling you exactly which extra to install.

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

## Quick Start

```python
from ImagePRO.pre_processing.blur import apply_average_blur
from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh
from ImagePRO.human_analysis.body_analysis.body_pose_estimation import detect_body_pose
from ImagePRO.object_analysis.object_detection import detect_objects
from ImagePRO.utils.image import Image

# Load an image
image = Image.from_path("person_and_objects.jpg")
# Or from numpy array: image = Image.from_array(np_array)

# Apply average blur
blur_result = apply_average_blur(image=image)
blur_result.save_as_img("blurred_output.jpg")

# Analyze face mesh (468 landmarks per detected face)
face_mesh_result = analyze_face_mesh(image=image)
print(f"Detected {len(face_mesh_result.data)} face(s)")
face_mesh_result.save_as_csv("face_landmarks.csv")

# Detect body pose (33 landmarks)
body_pose_result = detect_body_pose(image=image)
print(f"Body pose data: {body_pose_result.data}")

# Detect objects with YOLO
object_detection_result = detect_objects(
    image=image,
    accuracy_level=3  # 1=nano, 2=small, 3=medium, 4=large, 5=extra-large
)
print(f"Detected {len(object_detection_result.data)} objects")
object_detection_result.save_as_img("detections.jpg")
```

> **Note**: These are basic examples. Each module contains many more functions with extensive customization options. Explore the module-specific README files for detailed documentation.

## Documentation

Each module includes comprehensive documentation with detailed examples:

- **[Pre-processing](src/ImagePRO/pre_processing/README.md)**: Image manipulation, filtering, and enhancement
- **[Human Analysis](src/ImagePRO/human_analysis/README.md)**: Face and body analysis tools
  - [Face Analysis](src/ImagePRO/human_analysis/face_analysis/README.md): Face mesh, pose estimation, eye status, comparison
  - [Body Analysis](src/ImagePRO/human_analysis/body_analysis/README.md): Body pose and hand tracking
- **[Object Analysis](src/ImagePRO/object_analysis/README.md)**: YOLO-based object detection
- **[Utils](src/ImagePRO/utils/README.md)**: Shared utilities and I/O handling

For detailed project structure and development guidelines, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Architecture

ImagePRO is built with a modular architecture designed for extensibility and maintainability:

- **Clean Separation of Concerns**: Each module handles a specific domain
- **Consistent API Patterns**: All functions follow the same input/output conventions
- **Shared Utilities**: Common `Image` and `Result` classes for unified I/O
- **Lazy Optional Imports**: Heavy AI dependencies (MediaPipe, Ultralytics, InsightFace) are only imported inside the functions that use them, keeping `import ImagePRO` fast and independent of installed extras
- **Professional Error Handling**: Comprehensive validation with clear error messages
- **Type Safety**: Full type hints throughout the codebase
- **Documentation**: Google-style docstrings for all functions

### Key Design Principles
- **Non-Destructive Operations**: Operations never write into the input image; they return results in new `Result` objects (note: `crop_image` returns a view into the source array)
- **Functional Style**: Stateless functions that can be easily composed
- **Result Objects**: Unified return type containing image, data, and metadata
- **Keyword Arguments**: All optional parameters use keyword-only syntax

## Testing

The test suite uses pytest:

```bash
# Install with test tooling (editable install + dev extra)
pip install -e ".[dev]"

# Run the suite (optional heavy dependencies are mocked; base deps are enough)
pytest

# Include performance smoke guards
pytest --run-performance
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -r requirements/full.txt`
4. Install for development with test tooling: `pip install -e ".[dev]"`
5. Follow the coding standards outlined in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
6. Add tests for new features and run `pytest`
7. Update documentation as needed

### Reporting Issues
If you encounter any bugs or have feature requests, please open an issue on the GitHub repository with:
- Description of the problem or feature request
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
