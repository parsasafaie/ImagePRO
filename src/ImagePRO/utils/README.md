# Utils Module

Shared data structures for ImagePRO library.
Provides lightweight, immutable image wrappers and a unified result container with built-in save helpers.

## ✨ Features

- **Immutable Image Wrapper**: Lightweight `Image` class with factory constructors
- **Unified Result Object**: `Result` class to store images, data, and metadata
- **Built-in Saving**: Simple methods to save images and CSV files directly
- **Consistent API**: Designed for fluent pipelines and functional programming style

## 📚 Available Classes

### **Image**
Lightweight wrapper around `numpy.ndarray` with fluent, immutable transforms.  
Always use factory constructors to create instances.

#### **Factory Methods**
- **`Image.from_path(path)`** – Load an image from disk (BGR format by default).  
- **`Image.from_array(array, colorspace="BGR")`** – Wrap an existing `numpy.ndarray` as an image.

#### **Introspection**
- **`shape`** → Returns image shape (`H×W×C` or `H×W`)  
- **`dtype`** → Returns numpy dtype of underlying image  

### **Result**
Unified container for outputs of ImagePRO operations.
Holds optional image(s), structured data, and arbitrary metadata.

#### **Key Properties**
- **`image`** → A single np.ndarray or list of arrays
- **`data`** → Any structured data (e.g., landmark points)
- **`meta`** → Dictionary of metadata (e.g., processing parameters)

#### **Methods**
- **`save_as_img(path)`** – Save image(s) to disk (single file or auto-suffixed list).
- **`save_as_csv(path, rows=None)`** – Save structured data to a CSV file. Uses data by default.

## 🚀 Quick Start
```python
from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

image = Image.from_path('input.jpg') # Or -> image = Image.from_array(np_image)

result = analyze_face_mesh(image=image)

print(type(image))  # <class 'ImagePRO.utils.image.Image'>
print(type(result)) # <class 'ImagePRO.utils.result.Result'>

print(image._data) # np.ndarray
print(image.source_type) # 'path' or 'array'
print(image.path) # 'input.jpg' or None
print(image.shape) # (H, W, C)

print(result.image)  # np.ndarray or List[np.ndarray]
print(result.data)   # Any other data like landmarks list
print(result.meta)   # Some additional info about process

result.save_as_img('output.jpg') # Save image to path
result.save_as_csv('landmarks.csv') # Save data to CSV
```
## 🔧 Conventions

- **Colorspace:** Images are assumed to be `BGR` unless explicitly specified
- **Immutable Design:** Methods return new objects instead of mutating originals
- **Automatic Directory Creation:** Save helpers create parent folders if needed
- **Return Values:**: Consistent return types across all operations

## ⚠️ Error Handling

- **`ValueError`**: Invalid parameters or conflicting inputs
- **`TypeError`**: Incorrect input types
- **`IOError`**: File read/write failures

## 📝 Technical Notes

- **OpenCV Compatible**: Fully compatible with OpenCV I/O (`cv2.imread`, `cv2.imwrite`)
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Performance Optimized**: Efficient file operations and data handling
- **Pipeline Ready:** Ideal for method-chaining or functional API designs
