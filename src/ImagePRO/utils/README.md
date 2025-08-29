# Utils Module

Shared utilities and helper functions for the ImagePRO library.

## ✨ Features

- **Centralized I/O Operations**: Unified image loading and saving
- **Data Format Handling**: Consistent CSV export functionality
- **Error Management**: Robust input validation and error handling
- **Cross-module Support**: Shared utilities used throughout the library

## 📚 Available Classes

### **IOHandler**
Central class for all image and data I/O operations.

#### **Image Operations**
- **`load_image()`**: Load images from file paths or numpy arrays
- **`save_image()`**: Save images to disk with format detection
- **`save_csv()`**: Export data to CSV format

#### **Key Methods**
```python
from ImagePRO.utils.io_handler import IOHandler

# Load image (file path or numpy array)
image = IOHandler.load_image(
    image_path="path/to/image.jpg",  # OR
    np_image=existing_array
)

# Save image
IOHandler.save_image(
    np_image=image_array,
    result_path="output.jpg"
)

# Save CSV data
IOHandler.save_csv(
    data=landmark_data,
    result_path="landmarks.csv"
)
```

## 🔧 I/O Conventions

- **Flexible Input**: Accepts both file paths and numpy arrays
- **Automatic Format Detection**: Image format inferred from file extension
- **Error Handling**: Comprehensive validation with clear error messages
- **Return Values**: Consistent return types across all operations

## ⚠️ Error Handling

- **`ValueError`**: Invalid parameters or conflicting inputs
- **`TypeError`**: Incorrect input types
- **`FileNotFoundError`**: File not found or inaccessible
- **`IOError`**: File read/write failures

## 📝 Technical Notes

- **OpenCV Compatible**: Designed for BGR image format
- **Memory Efficient**: Minimizes unnecessary array copies
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Performance Optimized**: Efficient file operations and data handling
