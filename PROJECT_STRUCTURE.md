# ImagePRO Project Structure

## Overview
ImagePRO is organized into logical modules that### Import Patterns

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

## Directory Structure
```
[src/ImagePRO/]
├── __init__.py
├── utils/                              # Shared utilities
│   ├── __init__.py
│   ├── image.py                        [base]
│   ├── result.py                       [base]
│   └── README.md
├── pre_processing/                     # Image pre processing tools
│   ├── __init__.py
│   ├── blur.py                         [base]
│   ├── contrast.py                     [base]
│   ├── crop.py                         [base]
│   ├── dataset_generator.py            [base]
│   ├── grayscale.py                    [base]
│   ├── resize.py                       [base]
│   ├── rotate.py                       [base]
│   ├── sharpen.py                      [base]
│   └── README.md
├── human_analysis/                     # Human analysis capabilities
│   ├── __init__.py
│   ├── face_analysis/                  # Face analysis tools
│   │   ├── __init__.py
│   │   ├── eye_status_analysis.py      [mediapipe]
│   │   ├── face_comparison.py          [insightface]
│   │   ├── face_detection.py           [mediapipe]
│   │   ├── face_mesh_analysis.py       [mediapipe]
│   │── head_pose_estimation.py         [mediapipe]
│   │   └── README.md
│   ├── body_analysis/                  # Body analysis tools
│   │   ├── __init__.py
│   │   ├── body_pose_estimation.py     [mediapipe]
│   │   ├── hand_tracking.py            [mediapipe]
│   │   └── README.md
│   └── README.md
└── object_analysis/                    # Object detection
    ├── __init__.py
    ├── object_detection.py             [yolo]
    └── README.md

```

## Module Dependencies

### Core Dependencies
#### Base Dependencies
- **OpenCV** (cv2): Image processing operations
- **NumPy**: Array operations and data handling
#### Optional Dependencies
- **MediaPipe**: Human analysis (face, body, hands)
- **Ultralytics**: YOLO object detection
- **InsightFace**: Advanced face analysis

### Internal Dependencies
- **utils.image**: Core Image class used by all modules
- **utils.result**: Core Result class used by all modules
- **pre_processing**: Uses utils.image and utils.result
- **human_analysis**: Uses utils.image, utils.result, and face_mesh_analysis as base
- **object_analysis**: Uses utils.image and utils.result

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

### External Imports
```python
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
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

## Future Enhancements

### Planned Features
- AI-powered image enhancement
- Background removal and segmentation
- Video processing capabilities
- Web interface for non-programmers
- Plugin system for extensibility

### Code Improvements
- Replace print statements with proper logging
- Add configuration file support
- Implement batch processing utilities
- Add progress indicators for long operations
- Consider async support for I/O operations

## Maintenance Notes

### Regular Tasks
- Update dependency versions in requirements.txt
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
