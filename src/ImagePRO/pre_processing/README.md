# Pre-processing Module

Professional image manipulation, filtering, and enhancement utilities for computer vision pipelines.

## ✨ Features

- **Consistent I/O**: Support for both `from_pth` and `from_array` (numpy arrays)
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
- **`resize.py`**: Resize images to specified dimensions
- **`crop.py`**: Crop images using coordinate-based selection
- **`rotate.py`**: Rotate images (90°, 180°, 270°, custom angles)
- **`histogram.py`**:  Display histogram of image channels (BGR, RGB, or grayscale)

### **Filtering & Enhancement**
- **`blur.py`**: Multiple blur algorithms (average, Gaussian, median, bilateral)
- **`sharpen.py`**: Sharpening filters (Laplacian, Unsharp Masking)
- **`contrast.py`**: Contrast enhancement (CLAHE, GHE, stretching)

### **Advanced Features**
- **`dataset_generator.py`**: Automated image capture with preprocessing pipeline

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

- **`ValueError`**: Invalid parameters (e.g., negative kernel sizes, invalid coordinates, invalid image input)
- **`TypeError`**: Incorrect input types
- **`IOError`**: File saving/loading failures
- **`RuntimeError`**: Webcam opening failure 

## 📝 Notes

- Functions are **pure and stateless** - safe to reuse in loops
- **Processing order** matters in pipelines - consider dependencies
- **Memory efficient** - operations performed in-place when possible
