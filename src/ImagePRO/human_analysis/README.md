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

## 🔧 I/O Conventions

- **Input**: A `Image` instance created by path or array
- **Output**: A `Result` instance contains image(np.ndarray), data(any other data like landmarks list) and meta(some additional info about process)
- **Live Mode**: Webcam functions with ESC key to exit

## 📊 Data Formats

### **Landmark Data**
- **Format**: `[id, landmark_index, x, y, z]`
- **Coordinates**: Normalized values [0, 1] from MediaPipe
- **Conversion**: Multiply by image width/height for pixel coordinates

### **Pose Data**
- **Format**: `[id, yaw, pitch]` for head pose
- **Units**: Proportional values for orientation

## ⚠️ Error Handling

- **`ValueError`**: Invalid parameters(like invalid image) or no landmarks detected
- **`TypeError`**: Incorrect input types
- **`RuntimeError`**: Webcam access failures
- **`FileNotFoundError`**: Image file not found

## 📝 Technical Notes

- **MediaPipe Integration**: Uses state-of-the-art detection models
- **Coordinate System**: Normalized coordinates for cross-platform compatibility
- **Performance**: Optimized for both static images and video streams
- **Confidence**: Configurable detection thresholds for accuracy vs. speed
