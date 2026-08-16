# Utils Module

Shared data structures for ImagePRO library.
Provides lightweight image wrappers and a unified result container with built-in save helpers.

## Features

- **Lightweight Image Wrapper**: `Image` class with factory constructors
- **Unified Result Object**: `Result` class to store images, data, and metadata
- **Built-in Saving**: Simple methods to save images and CSV files directly
- **Consistent API**: Designed for fluent pipelines and functional programming style

## Available Classes

### **Image**
Lightweight wrapper around `numpy.ndarray` with factory constructors.
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

## Quick Start
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
## Conventions

- **Colorspace:** Images are assumed to be `BGR` (OpenCV default) unless explicitly specified
- **Non-Destructive Operations:** Operations never write into the input image; they return results in new `Result` objects (note: `crop_image` returns a view into the source array)
- **Automatic Directory Creation:** Save helpers create parent folders if needed
- **Return Values:** Consistent return types across all operations
- **Type Safety:** Full type hints for better IDE support and error detection

## Error Handling

- **`ValueError`**: Invalid parameters or conflicting inputs
- **`TypeError`**: Incorrect input types
- **`IOError`**: File read/write failures

## Technical Notes

- **OpenCV Compatible**: Fully compatible with OpenCV I/O (`cv2.imread`, `cv2.imwrite`)
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Performance Optimized**: Efficient file operations and data handling
- **Pipeline Ready:** Ideal for method-chaining or functional API designs
- **Memory Efficient**: Lightweight wrappers with minimal overhead
- **Type Hints**: Full type annotations for better developer experience

## Related Modules

- See [Pre-processing Module](../pre_processing/README.md) for image manipulation functions
- See [Human Analysis Module](../human_analysis/README.md) for face/body analysis
- See [Object Analysis Module](../object_analysis/README.md) for object detection
