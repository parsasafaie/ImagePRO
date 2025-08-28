# ImagePRO

> **Professional & Modular Image Processing Library in Python**

**ImagePRO** is a clean, modular, and easy-to-use Python library for image processing tasks, built with OpenCV, Mediapipe, YOLO and designed to be extensible for developers.

Whether you're working on computer vision pipelines, preprocessing images for AI models, or simply automating batch image edits — **ImagePRO** gives you powerful tools with minimal effort.

## Features (So Far)

- Image I/O management
- Rotation (90°, 180°, 270°, custom angles)
- Resize & Crop
- Grayscale conversion
- Blur filters (average, Gaussian, median, bilateral)
- Contrast enhancement (CLAHE, GHE, stretching)
- Sharpening filters (Laplacian, Unsharp Masking)
- Dataset Generator
- Face mesh analyzer
- Face detector
- Head Position Estimator
- Eye Status analyzer
- Hnad tracker
- Body Position Estimator
- Object Detection

More features are being added regularly!

## Installation

### From PyPI
```bash
pip install ImagePRO-Python
```

### From Source
1. Clone the Repository:
```bash
git clone https://github.com/parsasafaie/ImagePRO.git
cd ImagePRO
```

2. Create a Virtual Environment (Recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

3. Install Dependencies:
```bash
pip install -r requirements.txt
```

## License 
This project is licensed under the MIT License – see the LICENSE file for details.
