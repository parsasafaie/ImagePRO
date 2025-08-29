# Face Analysis Module

Advanced facial landmark detection, pose estimation, and analysis using MediaPipe technology.

## ✨ Features

- **468-point Face Mesh**: Complete facial landmark detection with tessellation
- **Head Pose Estimation**: Yaw and pitch calculation from facial geometry
- **Eye Status Analysis**: Open/closed detection using Eye Aspect Ratio (EAR)
- **Face Comparison**: Identity matching with InsightFace embeddings
- **Face Cropping**: Automated face region extraction and cropping
- **Real-time Processing**: Live webcam analysis for all functions

## 🔧 I/O Conventions

- **Input**: Provide either `src_image_path` or `src_np_image` (BGR format)
- **Output**: Optional saving of annotated images and landmark data
- **Precedence**: If both inputs provided, `src_np_image` takes priority
- **Live Mode**: Webcam functions with ESC key to exit

## 📚 Available Functions

### **Core Analysis**
- **`face_mesh_analysis.py`**: 468-point facial landmark detection
- **`head_pose_estimation.py`**: Head orientation (yaw, pitch) estimation
- **`eye_status_analysis.py`**: Eye open/closed status detection
- **`face_detection.py`**: Face region detection and cropping
- **`face_comparison.py`**: Face identity matching and comparison

## 🚀 Quick Start

```python
from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh
from ImagePRO.human_analysis.face_analysis.head_pose_estimation import estimate_head_pose
from ImagePRO.human_analysis.face_analysis.eye_status_analysis import analyze_eye_status

# Detect facial landmarks
annotated, landmarks = analyze_face_mesh(
    src_image_path="face.jpg",
    output_image_path="mesh.jpg",
    output_csv_path="landmarks.csv"
)

# Estimate head pose
pose_data = estimate_head_pose(
    src_np_image=annotated,
    output_csv_path="pose.csv"
)

# Analyze eye status
is_open = analyze_eye_status(
    src_np_image=annotated,
    threshold=0.22
)
```

## 📊 Output Formats

### **Landmarks CSV**
- **Format**: `[face_id, landmark_index, x, y, z]`
- **Coordinates**: Normalized values [0, 1] from MediaPipe
- **Conversion**: Multiply by image width/height for pixel coordinates

### **Pose Data**
- **Format**: `[face_id, yaw, pitch]`
- **Units**: Proportional values (not degrees)
- **Range**: Yaw: left/right, Pitch: up/down

## ⚠️ Error Handling

- **`ValueError`**: Invalid inputs or no faces detected
- **`TypeError`**: Incorrect parameter types
- **`RuntimeError`**: Webcam access failures
- **`FileNotFoundError`**: Image file not found

## 📝 Technical Notes

- **MediaPipe Integration**: Uses 468-point facial mesh model
- **Coordinate System**: Normalized coordinates for cross-platform compatibility
- **Performance**: Optimized for both static images and video streams
- **Confidence**: Configurable detection thresholds for accuracy vs. speed
