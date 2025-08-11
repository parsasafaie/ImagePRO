# Pre-Processing

Utilities to prepare images for analysis, training, or enhancement.  
Includes resizing, cropping, rotation, grayscale conversion, blurring, sharpening, and contrast adjustments.

## Features
- Consistent I/O: `src_image_path` or `src_np_image` (ndarray), optional `output_image_path`
- Drop-in functions for common ops (no classes)
- Safe argument validation + clear errors
- Plays nicely with OpenCV BGR images

## I/O Conventions
- Provide either:
  - `src_image_path: str` **or**
  - `src_np_image: np.ndarray` (BGR). If both provided, `src_np_image` takes precedence.
- If `output_image_path` is given, the function saves to disk and prints a log message; it still returns the processed `np.ndarray`.
- Paths are not prefixed with `src.` in imports.

## Submodules & Functions
### `grayscale.py`
- `convert_to_grayscale(src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`

### `resize.py`
- `resize_image(new_size: tuple[int, int], src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`

### `crop.py`
- `crop_image(start_point: tuple[int,int], end_point: tuple[int,int], src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`

### `rotate.py`
- `rotate_image_90/180/270(src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`
- `rotate_image_custom(angle: float, scale: float = 1.0, src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`

### `blur.py`
- `apply_average_blur(kernel_size=(5,5), src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`
- `apply_gaussian_blur(kernel_size=(5,5), src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`
- `apply_median_blur(filter_size=5, src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`
- `apply_bilateral_blur(filter_size=9, sigma_color=75, sigma_space=75, src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`

### `contrast.py`
- `apply_histogram_equalization(src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`
- `apply_clahe_contrast(clip_limit=2.0, tile_grid_size=(8,8), src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`
- `apply_contrast_stretching(alpha: float, beta: int, src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`

### `sharpen.py`
- `apply_laplacian_sharpening(coefficient=3.0, src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`
- `apply_unsharp_masking(coefficient=1.0, src_image_path=None, src_np_image=None, output_image_path=None) -> np.ndarray`

## Quick Start
```python
from pre_processing.grayscale import convert_to_grayscale
from pre_processing.blur import apply_gaussian_blur
from pre_processing.resize import resize_image

gray = convert_to_grayscale(src_np_image=my_bgr)
blur = apply_gaussian_blur(src_np_image=gray, kernel_size=(5, 5), output_image_path="blur.jpg")
resized = resize_image(new_size=(640, 480), src_np_image=blur)
```

## Error Handling
- `ValueError` / `TypeError`: invalid arguments (e.g., kernel sizes, coordinates)
- From `IOHandler.load_image`:
  - `TypeError`, `FileNotFoundError`, `ValueError` (bad path, both inputs None, or load failure)
- From `IOHandler.save_image`:
  - `TypeError`, `IOError` (invalid path or write failure)

## Notes
- All operations expect **BGR** input (OpenCV default).
- Functions are pure & stateless; reuse them freely in pipelines.
