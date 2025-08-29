# Body Analysis Module

Advanced body pose estimation and hand tracking using MediaPipe technology.

## ✨ Features

- **Body Pose Estimation**: 33-point body landmark detection
- **Hand Tracking**: 21-point hand landmark analysis
- **Real-time Processing**: Live webcam analysis capabilities
- **Flexible Output**: Optional image annotation and CSV export
- **Performance Optimized**: Efficient processing for both images and video

## 🔧 I/O Conventions

- **Input**: Support for both file paths and numpy arrays
- **Output**: Optional saving of annotated images and landmark data
- **Precedence**: Numpy arrays take priority when both inputs provided
- **Live Mode**: Webcam functions with ESC key to exit

## 📚 Available Functions

### **Body Analysis**
- **`body_pose_estimation.py`**: Full body pose detection (33 landmarks)
- **`hand_tracking.py`**: Hand landmark detection and tracking (21 points)

## 🚀 Quick Start

```python
from ImagePRO.human_analysis.body_analysis.body_pose_estimation import detect_body_pose
from ImagePRO.human_analysis.body_analysis.hand_tracking import detect_hands

# Detect body pose
annotated_body, body_landmarks = detect_body_pose(
    src_image_path="person.jpg",
    output_image_path="pose.jpg",
    output_csv_path="body_landmarks.csv"
)

# Track hands
annotated_hands, hand_landmarks = detect_hands(
    src_np_image=annotated_body,
    max_hands=2,
    output_image_path="hands.jpg"
)
```

## 📊 Output Formats

### **Body Landmarks CSV**
- **Format**: `[landmark_index, x, y, z]`
- **Coordinates**: Normalized values [0, 1] from MediaPipe
- **Total Points**: 33 body landmarks

### **Hand Landmarks CSV**
- **Format**: `[hand_id, landmark_index, x, y, z]`
- **Coordinates**: Normalized values [0, 1] from MediaPipe
- **Total Points**: 21 hand landmarks per hand

## ⚠️ Error Handling

- **`ValueError`**: Invalid parameters or no landmarks detected
- **`TypeError`**: Incorrect input types
- **`RuntimeError`**: Webcam access failures
- **`FileNotFoundError`**: Image file not found

## 📝 Technical Notes

- **MediaPipe Integration**: Uses state-of-the-art pose and hand models
- **Coordinate System**: Normalized coordinates for cross-platform compatibility
- **Performance**: Optimized for both static images and video streams
- **Multi-person Support**: Configurable for single or multiple subjects
