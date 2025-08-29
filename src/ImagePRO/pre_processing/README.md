# Pre-processing Module

Professional image manipulation, filtering, and enhancement utilities for computer vision pipelines.

## ✨ Features

- **Consistent I/O**: Support for both `src_image_path` and `src_np_image` (numpy arrays)
- **Drop-in Functions**: Simple, stateless operations with no class instantiation required
- **Safe Validation**: Comprehensive argument validation with clear error messages
- **OpenCV Compatible**: Designed to work seamlessly with OpenCV BGR images
- **Batch Support**: Functions can be easily chained for complex processing pipelines

## 🔧 I/O Conventions

- **Input**: Provide either:
  - `src_image_path: str` **OR**
  - `src_np_image: np.ndarray` (BGR format)
- **Output**: If `output_image_path` is given, the function saves to disk and prints a log message
- **Return**: All functions return the processed `np.ndarray` for further processing
- **Precedence**: If both inputs are provided, `src_np_image` takes precedence

## 📚 Available Functions

### **Basic Operations**
- **`grayscale.py`**: Convert images to single-channel grayscale
- **`resize.py`**: Resize images to specified dimensions
- **`crop.py`**: Crop images using coordinate-based selection
- **`rotate.py`**: Rotate images (90°, 180°, 270°, custom angles)

### **Filtering & Enhancement**
- **`blur.py`**: Multiple blur algorithms (average, Gaussian, median, bilateral)
- **`sharpen.py`**: Sharpening filters (Laplacian, Unsharp Masking)
- **`contrast.py`**: Contrast enhancement (CLAHE, GHE, stretching)

### **Advanced Features**
- **`dataset_generator.py`**: Automated image capture with preprocessing pipeline

## 🚀 Quick Start

```python
from ImagePRO.pre_processing.grayscale import convert_to_grayscale
from ImagePRO.pre_processing.blur import apply_gaussian_blur
from ImagePRO.pre_processing.resize import resize_image

# Load and convert to grayscale
gray = convert_to_grayscale(src_image_path="input.jpg")

# Apply Gaussian blur
blurred = apply_gaussian_blur(
    src_np_image=gray, 
    kernel_size=(5, 5)
)

# Resize the result
final = resize_image(
    new_size=(640, 480), 
    src_np_image=blurred,
    output_image_path="processed.jpg"
)
```

## ⚠️ Error Handling

- **`ValueError`**: Invalid parameters (e.g., negative kernel sizes, invalid coordinates)
- **`TypeError`**: Incorrect input types
- **`FileNotFoundError`**: Image file not found
- **`IOError`**: File saving/loading failures

## 📝 Notes

- All operations expect **BGR** input (OpenCV default)
- Functions are **pure and stateless** - safe to reuse in loops
- **Processing order** matters in pipelines - consider dependencies
- **Memory efficient** - operations performed in-place when possible
