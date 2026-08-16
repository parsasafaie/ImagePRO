"""Reusable fakes for MediaPipe/InsightFace/Ultralytics objects.

These are imported by test modules to inject deterministic detectors via
monkeypatch, keeping the human/object analysis tests hermetic (no model
downloads, no cameras, no GPU).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def make_landmarks(points_per_face, num_faces=1, total_points=478):
    """Build fake ``multi_face_landmarks``-style detections.

    points_per_face: list of (idx, x, y, z) tuples per face. Only the
    referenced indices need to exist, but a full-size landmark list is
    created so arbitrary indices resolve.
    """
    faces = []
    for face_points in points_per_face:
        landmark = [SimpleNamespace(x=0.0, y=0.0, z=0.0)] * total_points
        for idx, x, y, z in face_points:
            landmark[idx] = SimpleNamespace(x=x, y=y, z=z)
        faces.append(SimpleNamespace(landmark=landmark))
    return faces


class FakeFaceMesh:
    """Stand-in for mediapipe.solutions.face_mesh.FaceMesh."""

    def __init__(self, detection_result=None, **kwargs):
        self.kwargs = kwargs
        self.processed_images = []
        # Sentinel so tests notice when no detection result was wired up.
        self._detection_result = (
            detection_result
            if detection_result is not None
            else SimpleNamespace(multi_face_landmarks=None)
        )

    def process(self, image):
        self.processed_images.append(image)
        return self._detection_result


class FakePose:
    """Stand-in for mediapipe.solutions.pose.Pose."""

    def __init__(self, detection_result=None, **kwargs):
        self.kwargs = kwargs
        self.processed_images = []
        self._detection_result = (
            detection_result
            if detection_result is not None
            else SimpleNamespace(pose_landmarks=None)
        )

    def process(self, image):
        self.processed_images.append(image)
        return self._detection_result


class FakeHands:
    """Stand-in for mediapipe.solutions.hands.Hands."""

    def __init__(self, detection_result=None, **kwargs):
        self.kwargs = kwargs
        self.processed_images = []
        self._detection_result = (
            detection_result
            if detection_result is not None
            else SimpleNamespace(multi_hand_landmarks=None)
        )

    def process(self, image):
        self.processed_images.append(image)
        return self._detection_result


class FakeFaceAnalysisApp:
    """Stand-in for insightface.app.FaceAnalysis."""

    def __init__(self, faces_per_image):
        self.faces_per_image = faces_per_image
        self.prepared = False
        self.calls = 0

    def prepare(self, ctx_id):
        self.prepared = True

    def get(self, image):
        self.calls += 1
        faces = self.faces_per_image[min(self.calls - 1, len(self.faces_per_image) - 1)]
        return [
            SimpleNamespace(embedding=np.array(emb, dtype=np.float32)) for emb in faces
        ]


class FakeYOLOBox:
    """Mimics the ultralytics box attributes used by detect_objects."""

    def __init__(self, class_id, confidence, xyxyn):
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxyn = np.array([xyxyn])


class FakeYOLOResult:
    """Mimics the ultralytics Results attributes used by detect_objects."""

    def __init__(self, boxes, plotted_image=None):
        self.boxes = boxes
        self._plotted = (
            plotted_image
            if plotted_image is not None
            else np.zeros((8, 8, 3), np.uint8)
        )
        self.show_called = False

    def plot(self):
        return self._plotted

    def show(self):
        self.show_called = True


class FakeYOLO:
    """Stand-in for ultralytics.YOLO."""

    def __init__(self, results=None, model=None):
        self.model = model
        self._results = results if results is not None else []
        self.calls = []

    def __call__(self, image):
        self.calls.append(image)
        return list(self._results)
