# Body Analysis Module

Advanced body pose estimation and hand tracking using MediaPipe technology.

## ✨ Features

- **Body Pose Estimation**: 33-point body landmark detection
- **Hand Tracking**: 21-point hand landmark analysis
- **Real-time Processing**: Live webcam analysis capabilities
- **Flexible Output**: Optional image annotation and CSV export
- **Performance Optimized**: Efficient processing for both images and video

## 🔧 I/O Conventions

- **Input**: A `Image` instance created by path or array
- **Output**: A `Result` instance contains image(np.ndarray), data(any other data like landmarks list) and meta(some additional info about process)
- **Live Mode**: Webcam functions with ESC key to exit

## 📚 Available Functions

### **Body Analysis**
- **`body_pose_estimation.py`**: Full body pose detection (33 landmarks)
- **`hand_tracking.py`**: Hand landmark detection and tracking (21 points)

## 🚀 Quick Start

```python
from ImagePRO.human_analysis.body_analysis.body_pose_estimation import detect_body_pose
from ImagePRO.utils.image import Image

import cv2

image = Image.from_path('input.jpg') # Or -> image = Image.from_array(np_image)

# Detect body pose
body_pose_result = detect_body_pose(
    image=image,
    landmarks_idx=[i*5 for i in range(5)] # *5 to get 5 landmarks (0, 5, 10, 15, 20) in different points of body
)

# Better to use pre-loaded model for faster processing and lower memory usage (see docstring of detect_body_pose)

print(type(body_pose_result))  # Should print: <class 'ImagePRO.utils.result.Result'>
print(body_pose_result.data)  # Should print a list of 5 landmark info
print(body_pose_result.meta)  # Should print metadata about the operation

cv2.imshow("Landmark Image", body_pose_result.image) # Display the result image
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 📊 Data Formats

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

- **MediaPipe Integration**: Uses state-of-the-art pose and hand models from Google
- **Coordinate System**: Normalized coordinates [0, 1] for cross-platform compatibility
- **Performance**: Optimized for both static images and video streams
- **Multi-person Support**: Configurable for single or multiple subjects
- **Model Reuse**: Pre-load models for faster processing in loops (see function docstrings)
- **Real-time Processing**: Live webcam functions available for both pose and hand tracking

## 🔗 Related Modules

- See [Face Analysis](../face_analysis/README.md) for facial landmark detection
- See [Utils Module](../../utils/README.md) for `Image` and `Result` class documentation
- See [Pre-processing Module](../../pre_processing/README.md) for image manipulation
