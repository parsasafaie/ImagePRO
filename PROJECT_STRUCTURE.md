# ImagePRO Project Structure

## Overview
ImagePRO is organized into logical modules that provide specific functionality while maintaining clean separation of concerns.

## Directory Structure
```
src/ImagePRO/
├── __init__.py                 # Main package initialization
├── utils/                      # Shared utilities
│   ├── __init__.py
│   └── io_handler.py          # Image I/O operations
├── pre_processing/             # Image manipulation and enhancement
│   ├── __init__.py
│   ├── blur.py                # Blur filters (average, Gaussian, median, bilateral)
│   ├── contrast.py            # Contrast enhancement (CLAHE, GHE, stretching)
│   ├── crop.py                # Image cropping
│   ├── dataset_generator.py   # Automated image capture
│   ├── grayscale.py           # Grayscale conversion
│   ├── resize.py              # Image resizing
│   ├── rotate.py              # Image rotation
│   ├── sharpen.py             # Sharpening filters
│   └── README.md              # Module documentation
├── human_analysis/            # Human analysis capabilities
│   ├── __init__.py
│   ├── face_analysis/         # Face analysis tools
│   │   ├── __init__.py
│   │   ├── eye_status_analysis.py
│   │   ├── face_comparison.py
│   │   ├── face_detection.py
│   │   ├── face_mesh_analysis.py
│   │   ├── head_pose_estimation.py
│   │   └── README.md
│   ├── body_analysis/         # Body analysis tools
│   │   ├── __init__.py
│   │   ├── body_pose_estimation.py
│   │   ├── hand_tracking.py
│   │   └── README.md
│   └── README.md
└── object_analysis/           # Object detection
    ├── __init__.py
    ├── object_detection.py    # YOLO-based detection
    └── README.md
```

## Module Dependencies

### Core Dependencies
- **OpenCV** (cv2): Image processing operations
- **NumPy**: Array operations and data handling
- **MediaPipe**: Human analysis (face, body, hands)
- **Ultralytics**: YOLO object detection
- **InsightFace**: Advanced face analysis

### Internal Dependencies
- **utils.io_handler**: Used by all modules for I/O operations
- **pre_processing**: Independent, no internal dependencies
- **human_analysis**: Uses utils.io_handler
- **object_analysis**: Uses utils.io_handler

## Code Standards

### Constants
- All magic numbers are replaced with named constants
- Constants are defined at the top of each module
- Use UPPER_CASE naming convention

### Function Signatures
- Consistent parameter naming across all modules
- Support for both file paths and numpy arrays
- Optional output parameters for saving results

### Error Handling
- Comprehensive input validation
- Clear error messages
- Proper exception types (ValueError, TypeError, etc.)

### Documentation
- Detailed docstrings for all functions
- Parameter descriptions with types and defaults
- Return value documentation
- Exception documentation

## Import Patterns

### Internal Imports
```python
# Add parent directory to path for custom module imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_handler import IOHandler
```

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
3. Use `utils.io_handler` for I/O operations
4. Add comprehensive documentation
5. Include constants for configurable values

### Adding New Functions
1. Follow existing function signature patterns
2. Include comprehensive docstrings
3. Add input validation
4. Use consistent error handling
5. Support both file paths and numpy arrays

### Testing
- Test with both file paths and numpy arrays
- Verify error handling with invalid inputs
- Check output saving functionality
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
- All modules follow PEP 8 style guidelines
- Consistent error handling patterns
- Comprehensive input validation
- Clear and maintainable code structure
- Professional-grade documentation
