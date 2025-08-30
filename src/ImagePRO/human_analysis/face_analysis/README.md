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

- **Input**: A `Image` instance created by path or array
- **Output**: A `Result` instance contains image(np.ndarray), data(any other data like landmarks list) and meta(some additional info about process)
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
from ImagePRO.utils.image import Image

import cv2

image = Image.from_path('input.jpg') # Or -> image = Image.from_array(np_image)

# Analyze face mesh
mesh_result = analyze_face_mesh(
    image=image,
    landmarks_idx=[i*5 for i in range(5)] # *5 to get 5 landmarks (0, 5, 10, 15, 20) in different points of face
)

# Better to use pre-loaded model for faster processing and lower memory usage (see docstring of analyze_face_mesh)

print(type(mesh_result))  # Should print: <class 'ImagePRO.utils.result.Result'>
print(mesh_result.data)  # Should print a list of 5 landmark info
print(mesh_result.meta)  # Should print metadata about the operation

cv2.imshow("Landmark Image", mesh_result.image) # Display the result image
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 📊 Data Formats

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
