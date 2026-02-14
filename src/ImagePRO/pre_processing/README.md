# Pre-processing Module

Professional image manipulation, filtering, and enhancement utilities for computer vision pipelines.

## ✨ Features

- **Consistent I/O**: Support for both `from_path` and `from_array` (numpy arrays)
- **Drop-in Functions**: Simple, stateless operations with no class instantiation required
- **Safe Validation**: Comprehensive argument validation with clear error messages
- **OpenCV Compatible**: Designed to work seamlessly with OpenCV BGR images
- **Batch Support**: Functions can be easily chained for complex processing pipelines

## 🔧 I/O Conventions

- **Input**: A `Image` instance created by path or array
- **Output**: A `Result` instance contains image(np.ndarray), data(any other data like landmarks list) and meta(some additional info about process)
- **Return**: All functions return `Result` instance

## 📚 Available Functions

### **Basic Operations**
- **`grayscale.py`**: Convert images to single-channel grayscale
- **`resize.py`**: Resize images to specified dimensions (maintains aspect ratio or custom)
- **`crop.py`**: Crop images using coordinate-based selection (x, y, width, height)
- **`rotate.py`**: Rotate images (90°, 180°, 270°, or custom angles with optional scaling)
- **`histogram.py`**: Display histogram of image channels (BGR, RGB, or grayscale)

### **Filtering & Enhancement**
- **`blur.py`**: Multiple blur algorithms
  - `apply_average_blur`: Simple averaging filter
  - `apply_gaussian_blur`: Gaussian smoothing
  - `apply_median_blur`: Noise reduction
  - `apply_bilateral_blur`: Edge-preserving blur
- **`sharpen.py`**: Sharpening filters
  - `apply_laplacian_sharpening`: Edge enhancement
  - `apply_unsharp_masking`: Advanced sharpening technique
- **`contrast.py`**: Contrast enhancement
  - `apply_clahe_contrast`: Adaptive histogram equalization
  - `apply_histogram_equalization`: Global histogram equalization
  - `apply_contrast_stretching`: Linear contrast adjustment

### **Advanced Features**
- **`dataset_generator.py`**: Automated image capture with preprocessing pipeline
  - Webcam-based face dataset generation
  - Configurable preprocessing steps (blur, sharpen, grayscale, resize, rotate)
  - Automatic face detection and cropping

## 🚀 Quick Start

```python
from ImagePRO.pre_processing.blur import apply_gaussian_blur
from ImagePRO.utils.image import Image

import cv2

image = Image.from_path("input.jpg") # Or -> image = Image.from_array(np_image)

# Load and convert to grayscale
blurred_result = apply_gaussian_blur(image=image, kernel_size=(5, 5))

print(type(blurred_result))  # Should print: <class 'ImagePRO.utils.result.Result'>
print(blurred_result.data)  # Should print: None (as no extra data is returned in this function)
print(blurred_result.meta)  # Should print metadata about the operation

cv2.imshow("Blurred Image", blurred_result.image) # Display the blurred image
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## ⚠️ Error Handling

All functions follow consistent error handling patterns:

- **`ValueError`**: Invalid parameters (e.g., negative kernel sizes, invalid coordinates, invalid image input, out-of-range values)
- **`TypeError`**: Incorrect input types (e.g., passing string instead of Image object)
- **`IOError`**: File saving/loading failures
- **`RuntimeError`**: Webcam opening failure (for dataset_generator)

Error information is also stored in the `Result.meta` dictionary for programmatic access. 

## 📝 Notes

- Functions are **pure and stateless** - safe to reuse in loops and parallel processing
- **Processing order** matters in pipelines - consider dependencies (e.g., resize before crop)
- **Memory efficient** - operations performed efficiently with NumPy
- **OpenCV Compatible** - All functions work with BGR color space (OpenCV default)
- **Batch Processing** - Functions can be easily chained for complex pipelines

## 🔗 Related Modules

- See [Utils Module](../utils/README.md) for `Image` and `Result` class documentation
- See [Human Analysis Module](../human_analysis/README.md) for face/body analysis
- See [Object Analysis Module](../object_analysis/README.md) for object detection
