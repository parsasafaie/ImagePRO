# Human Analysis Module

Advanced human analysis capabilities including facial landmark detection, body pose estimation, and hand tracking using MediaPipe technology.

## ✨ Features

- **Comprehensive Face Analysis**: 468-point mesh, pose estimation, eye status
- **Full Body Tracking**: 33-point pose landmarks and gesture recognition
- **Hand Analysis**: 21-point hand landmarks for both hands
- **Real-time Processing**: Live webcam analysis for all modules
- **Flexible Output**: Optional image annotation and CSV export

## 📚 Submodules

### **Face Analysis**
Advanced facial analysis with multiple detection capabilities:
- **Face Mesh**: Complete 468-point facial landmark detection
- **Head Pose**: Yaw and pitch estimation from facial geometry
- **Eye Status**: Open/closed detection using Eye Aspect Ratio
- **Face Comparison**: Identity matching with InsightFace embeddings
- **Face Cropping**: Automated face region extraction

### **Body Analysis**
Full body pose estimation and hand tracking:
- **Body Pose**: 33-point body landmark detection
- **Hand Tracking**: 21-point hand landmark analysis
- **Multi-person Support**: Configurable for single or multiple subjects

## 🚀 Quick Start

```python
from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh
from ImagePRO.human_analysis.body_analysis.body_pose_estimation import detect_body_pose
from ImagePRO.human_analysis.body_analysis.hand_tracking import detect_hands

# Analyze face
face_result, face_landmarks = analyze_face_mesh(
    src_image_path="person.jpg",
    output_image_path="face_mesh.jpg"
)

# Detect body pose
body_result, body_landmarks = detect_body_pose(
    src_np_image=face_result,
    output_image_path="body_pose.jpg"
)

# Track hands
hands_result, hand_landmarks = detect_hands(
    src_np_image=body_result,
    max_hands=2,
    output_image_path="hands.jpg"
)
```

## 🔧 I/O Conventions

- **Input**: Support for both file paths and numpy arrays
- **Output**: Optional saving of annotated images and landmark data
- **Precedence**: Numpy arrays take priority when both inputs provided
- **Live Mode**: Webcam functions with ESC key to exit

## 📊 Output Formats

### **Landmark Data**
- **Format**: `[id, landmark_index, x, y, z]`
- **Coordinates**: Normalized values [0, 1] from MediaPipe
- **Conversion**: Multiply by image width/height for pixel coordinates

### **Pose Data**
- **Format**: `[id, yaw, pitch]` for head pose
- **Units**: Proportional values for orientation

## ⚠️ Error Handling

- **`ValueError`**: Invalid parameters or no landmarks detected
- **`TypeError`**: Incorrect input types
- **`RuntimeError`**: Webcam access failures
- **`FileNotFoundError`**: Image file not found

## 📝 Technical Notes

- **MediaPipe Integration**: Uses state-of-the-art detection models
- **Coordinate System**: Normalized coordinates for cross-platform compatibility
- **Performance**: Optimized for both static images and video streams
- **Confidence**: Configurable detection thresholds for accuracy vs. speed
