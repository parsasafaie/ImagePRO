"""Unit tests for face detection/cropping (analyze_face_mesh mocked)."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ImagePRO.human_analysis.face_analysis import face_detection
from ImagePRO.human_analysis.face_analysis.face_detection import detect_faces
from ImagePRO.utils.result import Result


def mesh_result_with_faces(faces_data):
    return Result(
        image=None,
        data=faces_data,
        meta={"operation": "analyze_face_mesh"},
    )


@pytest.fixture
def patch_mesh(monkeypatch):
    def _patch(faces_data):
        monkeypatch.setattr(
            face_detection, "analyze_face_mesh", lambda **kwargs: mesh_result_with_faces(faces_data)
        )

    return _patch


class TestDetectFaces:
    def test_crops_region_defined_by_outline(self, sample_bgr_array, sample_bgr_image, patch_mesh):
        # 100-wide, 80-tall image fixture is replaced here with exact sizes.
        height, width = sample_bgr_image.shape[:2]
        face_outline = [
            [0, 10, 0.2, 0.2, 0.0],
            [0, 33, 0.4, 0.4, 0.0],
        ]
        patch_mesh([face_outline])

        result = detect_faces(image=sample_bgr_image)

        expected_polygon = np.array(
            [[int(0.2 * width), int(0.2 * height)], [int(0.4 * width), int(0.4 * height)]],
            dtype=np.int32,
        )
        x, y, w, h = cv2.boundingRect(expected_polygon)
        assert isinstance(result.image, list)
        assert len(result.image) == 1
        assert np.array_equal(result.image[0], sample_bgr_array[y : y + h, x : x + w])

        assert result.data is not None
        assert np.array_equal(result.data[0], expected_polygon)

    def test_no_face_returns_none_with_error_meta(self, sample_bgr_image, patch_mesh):
        patch_mesh([])  # analyze_face_mesh returned falsy data
        result = detect_faces(image=sample_bgr_image)
        assert result.image is None
        assert result.data is None
        assert result.meta["error"] == "No face landmarks detected"

    def test_multiple_faces_produce_multiple_crops(self, sample_bgr_image, patch_mesh):
        faces = [
            [[0, 10, 0.1, 0.1, 0.0], [0, 33, 0.3, 0.3, 0.0]],
            [[1, 10, 0.5, 0.5, 0.0], [1, 33, 0.7, 0.7, 0.0]],
        ]
        patch_mesh(faces)
        result = detect_faces(image=sample_bgr_image, max_faces=2)
        assert isinstance(result.image, list)
        assert len(result.image) == 2
        assert len(result.data) == 2

    def test_meta_contents(self, sample_bgr_image, patch_mesh):
        patch_mesh([])
        result = detect_faces(image=sample_bgr_image, max_faces=2, min_confidence=0.9)
        assert result.meta["operation"] == "detect_faces"
        assert result.meta["max_faces"] == 2
        assert result.meta["min_confidence"] == 0.9
        assert result.meta["source"] is sample_bgr_image

    def test_mesh_called_with_outline_indices(self, sample_bgr_image, monkeypatch):
        captured = {}

        def fake_mesh(**kwargs):
            captured.update(kwargs)
            return Result(image=None, data=None, meta={})

        monkeypatch.setattr(face_detection, "analyze_face_mesh", fake_mesh)
        detect_faces(image=sample_bgr_image, max_faces=4, min_confidence=0.6)
        assert captured["landmarks_idx"] == face_detection.FACE_OUTLINE_INDICES
        assert captured["max_faces"] == 4
        assert captured["min_confidence"] == 0.6


class TestDetectFacesValidation:
    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            detect_faces(image=np.zeros((10, 10, 3)))

    @pytest.mark.parametrize("max_faces", [0, -2, 1.5, "2"])
    def test_invalid_max_faces_raises(self, sample_bgr_image, max_faces):
        with pytest.raises(ValueError):
            detect_faces(image=sample_bgr_image, max_faces=max_faces)

    @pytest.mark.parametrize("min_confidence", [-0.5, 1.5, "0.7"])
    def test_invalid_confidence_raises(self, sample_bgr_image, min_confidence):
        with pytest.raises(ValueError):
            detect_faces(image=sample_bgr_image, min_confidence=min_confidence)
