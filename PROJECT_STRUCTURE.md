# ImagePRO Project Structure

## Overview
ImagePRO is organized into logical modules that provide professional image processing capabilities. The library follows a modular architecture with clear separation of concerns, consistent API patterns, and comprehensive error handling.

## Directory Structure
```
ImagePRO/
├── LICENSE                              # MIT License
├── PROJECT_STRUCTURE.md                 # This file - project structure documentation
├── pyproject.toml                       # Python package configuration
├── readme.md                            # Main project README
├── requirements/                        # Dependency management
│   ├── base.txt                        # Core dependencies (OpenCV, NumPy)
│   ├── full.txt                        # All dependencies
│   ├── insightface.txt                 # InsightFace dependencies
│   ├── mediapipe.txt                   # MediaPipe dependencies
│   └── yolo.txt                        # YOLO/Ultralytics dependencies
└── src/
    └── ImagePRO/                       # Main package
        ├── __init__.py                 # Package initialization
        ├── utils/                      # Shared utilities
        │   ├── __init__.py
        │   ├── image.py                [base] - Image wrapper class
        │   ├── result.py               [base] - Result container class
        │   └── README.md
        ├── pre_processing/             # Image preprocessing tools
        │   ├── __init__.py
        │   ├── blur.py                 [base] - Blur filters
        │   ├── contrast.py             [base] - Contrast enhancement
        │   ├── crop.py                 [base] - Image cropping
        │   ├── dataset_generator.py    [base+mediapipe] - Dataset generation
        │   ├── grayscale.py            [base] - Grayscale conversion
        │   ├── histogram.py            [base] - Histogram visualization
        │   ├── resize.py               [base] - Image resizing
        │   ├── rotate.py               [base] - Image rotation
        │   ├── sharpen.py              [base] - Sharpening filters
        │   └── README.md
        ├── human_analysis/             # Human analysis capabilities
        │   ├── __init__.py
        │   ├── face_analysis/          # Face analysis tools
        │   │   ├── __init__.py
        │   │   ├── eye_status_analysis.py    [mediapipe] - Eye open/closed detection
        │   │   ├── face_comparison.py        [insightface] - Face identity matching
        │   │   ├── face_detection.py         [mediapipe] - Face detection & cropping
        │   │   ├── face_mesh_analysis.py     [mediapipe] - 468-point face mesh
        │   │   ├── head_pose_estimation.py   [mediapipe] - Head pose (yaw, pitch)
        │   │   └── README.md
        │   ├── body_analysis/          # Body analysis tools
        │   │   ├── __init__.py
        │   │   ├── body_pose_estimation.py   [mediapipe] - 33-point body pose
        │   │   ├── hand_tracking.py          [mediapipe] - 21-point hand landmarks
        │   │   └── README.md
        │   └── README.md
        └── object_analysis/            # Object detection
            ├── __init__.py
            ├── object_detection.py     [yolo] - YOLO object detection
            └── README.md
```

**Legend:**
- `[base]` - Requires only base dependencies
- `[mediapipe]` - Requires MediaPipe dependency
- `[yolo]` - Requires Ultralytics YOLO dependency
- `[insightface]` - Requires InsightFace dependency

## Module Dependencies

### Core Dependencies
#### Base Dependencies
- **OpenCV** (cv2): Image processing operations
- **NumPy**: Array operations and data handling
- **Matplotlib**: Data visualization and plotting

#### Optional Dependencies
- **MediaPipe**: Human analysis (face, body, hands)
- **Ultralytics**: YOLO object detection
- **InsightFace**: Advanced face analysis

### Internal Dependencies
- **utils.image**: Core `Image` class used by all modules for input handling
- **utils.result**: Core `Result` class used by all modules for output handling
- **pre_processing**: Uses `utils.image` and `utils.result`; independent module
- **human_analysis**: 
  - Uses `utils.image` and `utils.result`
  - Face analysis modules may depend on `face_mesh_analysis` for base functionality
  - Body analysis modules are independent
- **object_analysis**: Uses `utils.image` and `utils.result`; independent module

## Code Standards

### Constants
- All magic numbers are replaced with named constants
- Constants are defined at the top of each module
- Use UPPER_CASE naming convention

### Function Signatures
- Consistent parameter naming across all modules
- Standard Image class input parameter
- Standard Result class return type
- Keyword-only arguments with defaults
- Type hints for all parameters and returns

### Error Handling
- Comprehensive input validation
- Separation of TypeError and ValueError
- Clear, descriptive error messages
- Result object with error information in meta
- Consistent error patterns across modules

### Documentation
- Detailed docstrings for all functions
- Parameter descriptions with types and defaults
- Return value documentation
- Exception documentation

## Import Patterns

### Standard Import Order
```python
from __future__ import annotations  # Always first

# Python standard library
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Third-party libraries
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Local imports
from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result
```

## Development Guidelines

### Adding New Modules
1. Create directory with `__init__.py`
2. Follow existing naming conventions
3. Use `Result` and `Image` for I/O operations
4. Add comprehensive documentation
5. Include constants for configurable values

### Adding New Functions
1. Use Google-style docstrings with Args/Returns/Raises
2. Accept Image class input and return Result class
3. Use keyword-only arguments with type hints
4. Implement proper error validation with TypeError/ValueError
5. Follow consistent error handling patterns with Result.meta

### Testing
- Test with both Image.from_path and Image.from_array
- Test Result object contents and metadata
- Verify type and value error handling
- Test error information in Result.meta
- Test edge cases and boundary conditions

## Module Details

### Utils Module
Core utilities that provide the foundation for all other modules:
- **Image**: Immutable wrapper around numpy arrays with factory constructors
- **Result**: Unified container for operation outputs (image, data, metadata)

### Pre-processing Module
Image manipulation and enhancement functions:
- **Basic Operations**: Resize, crop, rotate, grayscale conversion
- **Filtering**: Multiple blur algorithms and sharpening techniques
- **Enhancement**: Contrast adjustment methods (CLAHE, histogram equalization, stretching)
- **Utilities**: Histogram visualization and dataset generation

### Human Analysis Module
Advanced human detection and analysis:
- **Face Analysis**: 468-point mesh, head pose, eye status, face comparison, detection
- **Body Analysis**: 33-point pose estimation and 21-point hand tracking
- **Real-time Support**: Live webcam processing for all analysis functions

### Object Analysis Module
YOLO-based object detection:
- **Multiple Models**: Nano to extra-large accuracy levels
- **Flexible**: Pre-trained or custom model support
- **Efficient**: GPU acceleration when available

## Future Enhancements

### Planned Features
- AI-powered image enhancement
- Background removal and segmentation
- Video processing capabilities
- Web interface for non-programmers
- Plugin system for extensibility
- Additional preprocessing filters
- Advanced face recognition features

### Code Improvements
- Replace print statements with proper logging
- Add configuration file support
- Implement batch processing utilities
- Add progress indicators for long operations
- Consider async support for I/O operations
- Enhanced error messages with troubleshooting tips

## Maintenance Notes

### Regular Tasks
- Update dependency versions in requirements files
- Verify all __init__.py files are properly configured
- Check for consistent code formatting
- Update documentation for new features
- Review and update constants as needed

### Code Quality
- Follows PEP 8 style guidelines
- Uses type hints throughout codebase
- Consistent I/O with Image and Result classes
- Google-style docstrings for all functions
- Clear separation of concerns
- Proper error handling hierarchy
- Professional-grade documentation
