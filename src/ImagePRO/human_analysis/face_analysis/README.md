# Human Face Analysis Module

A comprehensive module for analyzing human facial features and behavior using MediaPipe and OpenCV.

This module supports tasks such as face detection, 3D facial landmark extraction, mesh-based tracking, and head pose estimation.


## Submodules

- **`face_detection.py`**  
  Detects human faces in images or video streams using bounding box-based approaches.  
  Useful for quickly locating faces before applying finer-grained analysis.

- **`face_mesh_analysis.py`**  
  Extracts 468 high-fidelity facial landmarks per detected face using MediaPipe FaceMesh.  
  Capabilities:
  - Visualize full face mesh or specific landmark indices.
  - Output landmarks to CSV for further analysis.
  - Real-time overlay support.

- **`head_pose_estimation.py`**  
  Estimates head orientation using selected 3D facial landmarks:
  - `yaw` – rotation around the vertical axis (left ↔ right)
  - `pitch` – rotation around the lateral axis (up ↕ down)
  - (Optional extension: `roll` – tilt)


## Example Usage

```python
from human_analysis.face_analysis.head_pose_estimation import estimate_head_pose

pose = estimate_head_pose(np_image=img)
print(f"Yaw: {pose[0][1]}, Pitch: {pose[0][2]}")
```