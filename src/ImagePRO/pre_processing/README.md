# Pre-Processing Module

A collection of image preprocessing utilities to prepare images for analysis, training, or enhancement tasks.

This module includes common preprocessing operations such as resizing, cropping, rotation, grayscale conversion, blurring, sharpening, and contrast adjustments.


## Submodules

- **`blur.py`**  
  Apply various smoothing techniques:
  - `apply_average_blur` – Box filter
  - `apply_gaussian_blur` – Gaussian filter
  - `apply_median_blur` – Reduces salt-and-pepper noise
  - `apply_bilateral_blur` – Edge-preserving smoothing

- **`contrast.py`**  
  Contrast enhancement operations:
  - `apply_histogram_equalization` – Global equalization
  - `apply_clahe_contrast` – Adaptive histogram equalization (CLAHE)
  - `apply_contrast_stretching` – Linear contrast & brightness scaling

- **`crop.py`**  
  - `crop_image` – Crop a rectangular region from an image using pixel coordinates

- **`grayscale.py`**  
  - `convert_to_grayscale` – Convert image to single-channel grayscale (BGR to Gray)

- **`resize.py`**  
  - `resize_image` – Resize image to target width and height

- **`rotate.py`**  
  Image rotation utilities:
  - `rotate_image_90` – Rotate 90° clockwise
  - `rotate_image_180` – Rotate 180°
  - `rotate_image_270` – Rotate 270° clockwise
  - `rotate_image_custom` – Rotate by arbitrary angle with optional scaling

- **`sharpen.py`**  
  Image sharpening filters:
  - `apply_laplacian_sharpening` – Enhance edges using Laplacian filter
  - `apply_unsharp_masking` – Subtract blur for edge emphasis (unsharp mask)


## Usage Example

```python
from pre_processing.grayscale import convert_to_grayscale
from pre_processing.blur import apply_gaussian_blur

gray = convert_to_grayscale(np_image=my_image)
blurred = apply_gaussian_blur(np_image=gray, kernel_size=(5, 5))
```
