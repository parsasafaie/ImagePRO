# Pipeline Module

A reusable, ordered chain of image processing operations.
Define a pipeline once from the functions you need, then run it on any input as many times as you want.

## Features

- **Define Once, Run Anywhere**: Build a pipeline from a list of functions and reuse it for many inputs
- **Flexible Inputs**: Run on an `Image`, a NumPy array, or a file path
- **Per-Step Arguments**: Configure each function's keyword arguments with `Step`
- **Fluent Building**: Append operations with `apply()`; pipelines are immutable and `apply()` returns a new one
- **Composable**: Works with every ImagePRO function and any custom callable following the Image → Result convention
- **Pipeline Metadata**: The returned `Result` lists every executed step and each step's own metadata

## Available Classes

### **Pipeline**
An ordered, immutable chain of operations. `run()` normalizes the input to an
`Image`, feeds each operation's output to the next one, and returns the final
`Result` with pipeline metadata (`operation`, `steps`, `step_meta`, `source`)
merged over the final operation's metadata.

#### **Constructor**
- **`Pipeline(operations)`** – Build from a list/tuple of callables or `Step` objects.
- **`Pipeline(*operations)`** – Or pass them as separate arguments.
- Passing another `Pipeline` as an operation nests it as a single step.

#### **Methods**
- **`run(image)`** – Execute all steps in order; also available as `pipeline(image)`.
- **`apply(function, **kwargs)`** – Return a new pipeline with one more step appended.

#### **Introspection**
- **`operations`** → Tuple of `Step` objects in execution order
- **`len(pipeline)`** → Number of steps
- **`iter(pipeline)`** → Iterate over `Step` objects

### **Step**
A single operation with its keyword arguments. Keyword arguments are forwarded
to the wrapped function on every run.

- **`Step(function, **kwargs)`** – e.g. `Step(resize_image, new_size=(800, 600))`
- **`name`** → Function name, used in pipeline metadata

## Quick Start
```python
from ImagePRO.pipeline import Pipeline, Step
from ImagePRO.pre_processing.blur import apply_gaussian_blur
from ImagePRO.pre_processing.grayscale import convert_to_grayscale
from ImagePRO.pre_processing.resize import resize_image

# Define the pipeline once (functions run top to bottom)
pipeline = Pipeline([
    Step(apply_gaussian_blur, kernel_size=(7, 7)),
    convert_to_grayscale,
    Step(resize_image, new_size=(800, 600)),
])

# Reuse it on any input: Image, numpy array, or file path
result = pipeline.run("photo.jpg")
result = pipeline.run(np_image_array)
result.save_as_img("processed.jpg")

print(result.meta["steps"])     # ['apply_gaussian_blur', 'convert_to_grayscale', 'resize_image']
print(result.meta["step_meta"]) # Per-step metadata dicts
print(len(pipeline))            # 3

# Fluent building (returns a new pipeline; the original is unchanged)
small = pipeline.apply(resize_image, new_size=(64, 64))
result = small.run("photo.jpg")

# Calling the pipeline directly is shorthand for run()
result = pipeline("photo.jpg")
```

## Custom Operations
Any callable that accepts an `Image` as its first positional argument works as
a step. Returning a `Result` is the convention; returning a plain
`numpy.ndarray` is also accepted and treated as the output image.

```python
import numpy as np
from ImagePRO.pipeline import Pipeline
from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


def threshold_dark_pixels(image: Image, *, max_value: int = 50) -> Result:
    mask = image._data.max(axis=2) <= max_value
    output = np.where(mask[..., None], 0, image._data)
    return Result(image=output, meta={"operation": "threshold_dark_pixels"})


pipeline = Pipeline([
    Step(threshold_dark_pixels, max_value=30),
    apply_gaussian_blur,
])
result = pipeline.run("photo.jpg")
```

## Notes
- Steps run in definition order; the output image of each step is wrapped in an
  `Image` and passed to the next step. 2D intermediates are tagged as GRAY.
- Only the **final** step may return a result without an image (e.g. a
  data-only result); otherwise a `ValueError` explains which step broke the chain.
- Operations raise their own `TypeError`/`ValueError` for invalid inputs, and
  these propagate unchanged (e.g. running `convert_to_grayscale` on an image
  that is already grayscale raises `ValueError`).
- `run()` never mutates the pipeline, so a single instance is safe to reuse
  across inputs.
