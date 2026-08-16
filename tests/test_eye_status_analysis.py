"""Unit tests for eye status analysis (analyze_face_mesh mocked)."""

from __future__ import annotations

import pytest

from ImagePRO.human_analysis.face_analysis import eye_status_analysis
from ImagePRO.human_analysis.face_analysis.eye_status_analysis import analyze_eye_status
from ImagePRO.utils.result import Result


def patch_mesh(monkeypatch, faces_data):
    monkeypatch.setattr(
        eye_status_analysis,
        "analyze_face_mesh",
        lambda **kwargs: Result(image=None, data=faces_data, meta={}),
    )


def eye_rows(x_left, x_right, y_top, y_bottom):
    # Landmark rows: [face_id, idx, x, y, z] for indices 386, 374, 263, 362.
    return [
        [0, 386, 0.0, y_top, 0.0],
        [0, 374, 0.0, y_bottom, 0.0],
        [0, 263, x_left, 0.0, 0.0],
        [0, 362, x_right, 0.0, 0.0],
    ]


class TestAnalyzeEyeStatus:
    def test_open_eye_returns_true(self, sample_bgr_image, monkeypatch):
        # vertical 0.1, horizontal 0.2 -> EAR = 0.5 > 0.2 threshold.
        patch_mesh(monkeypatch, eye_rows(x_left=0.2, x_right=0.4, y_top=0.3, y_bottom=0.4))
        result = analyze_eye_status(
            image=sample_bgr_image, face_mesh_obj=object()
        )
        assert result.data is True

    def test_closed_eye_returns_false(self, sample_bgr_image, monkeypatch):
        # vertical 0.0 -> EAR = 0.
        patch_mesh(monkeypatch, eye_rows(x_left=0.2, x_right=0.4, y_top=0.3, y_bottom=0.3))
        result = analyze_eye_status(image=sample_bgr_image, face_mesh_obj=object())
        assert result.data is False

    def test_custom_threshold(self, sample_bgr_image, monkeypatch):
        # EAR = 0.25: open with default 0.2, closed with 0.4.
        patch_mesh(monkeypatch, eye_rows(x_left=0.2, x_right=0.6, y_top=0.3, y_bottom=0.4))
        default = analyze_eye_status(image=sample_bgr_image, face_mesh_obj=object())
        strict = analyze_eye_status(
            image=sample_bgr_image, threshold=0.4, face_mesh_obj=object()
        )
        assert default.data is True
        assert strict.data is False

    def test_zero_horizontal_distance_reports_closed(self, sample_bgr_image, monkeypatch):
        patch_mesh(monkeypatch, eye_rows(x_left=0.3, x_right=0.3, y_top=0.3, y_bottom=0.4))
        result = analyze_eye_status(image=sample_bgr_image, face_mesh_obj=object())
        assert result.data is False

    def test_pixel_scale_uses_image_dimensions(self, monkeypatch):
        import numpy as np

        from ImagePRO.utils.image import Image

        # Non-square image: 40 wide, 20 tall. Distances scale by axis.
        image = Image.from_array(np.zeros((20, 40, 3), np.uint8), colorspace="BGR")
        patch_mesh(monkeypatch, eye_rows(x_left=0.0, x_right=0.5, y_top=0.0, y_bottom=0.5))
        # vertical = 0.5 * 20 = 10, horizontal = 0.5 * 40 = 20 -> EAR 0.5.
        result = analyze_eye_status(image=image, face_mesh_obj=object())
        assert result.data is True

    def test_no_face_returns_none_with_error(self, sample_bgr_image, monkeypatch):
        patch_mesh(monkeypatch, [])
        result = analyze_eye_status(image=sample_bgr_image, face_mesh_obj=object())
        assert result.data is None
        assert result.meta["error"] == "No face landmarks detected"

    def test_missing_landmark_returns_none_with_error(self, sample_bgr_image, monkeypatch):
        patch_mesh(monkeypatch, [[0, 999, 0.1, 0.1, 0.0]])
        result = analyze_eye_status(image=sample_bgr_image, face_mesh_obj=object())
        assert result.data is None
        assert "Missing landmark" in result.meta["error"]

    def test_meta_contents(self, sample_bgr_image, monkeypatch):
        patch_mesh(monkeypatch, eye_rows(0.2, 0.4, 0.3, 0.4))
        result = analyze_eye_status(
            image=sample_bgr_image,
            min_confidence=0.5,
            threshold=0.3,
            face_mesh_obj=object(),
        )
        assert result.meta["operation"] == "analyze_eye_status"
        assert result.meta["min_confidence"] == 0.5
        assert result.meta["threshold"] == 0.3


class TestAnalyzeEyeStatusValidation:
    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            analyze_eye_status(image="img")

    @pytest.mark.parametrize("min_confidence", [-0.1, 1.1, "0.5"])
    def test_invalid_confidence_raises(self, sample_bgr_image, min_confidence):
        with pytest.raises(ValueError):
            analyze_eye_status(image=sample_bgr_image, min_confidence=min_confidence)

    def test_non_numeric_confidence_raises(self, sample_bgr_image):
        with pytest.raises(TypeError):
            analyze_eye_status(image=sample_bgr_image, min_confidence=None)
