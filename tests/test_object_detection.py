"""Unit tests for YOLO object detection (ultralytics faked)."""

from __future__ import annotations

import numpy as np
import pytest

from ImagePRO.object_analysis.object_detection import detect_objects

from fakes import FakeYOLO, FakeYOLOBox, FakeYOLOResult


@pytest.fixture
def fake_yolo():
    result = FakeYOLOResult(
        boxes=[FakeYOLOBox(class_id=7, confidence=0.87, xyxyn=(0.1, 0.2, 0.3, 0.4))]
    )
    return FakeYOLO(results=[result])


class TestDetectObjects:
    def test_parses_detection_data(self, sample_bgr_image, fake_yolo):
        result = detect_objects(image=sample_bgr_image, model=fake_yolo)
        assert result.data == [[7, [0.1, 0.2, 0.3, 0.4], 0.87]]

    def test_no_detections_returns_empty_list(self, sample_bgr_image):
        result = detect_objects(
            image=sample_bgr_image, model=FakeYOLO(results=[FakeYOLOResult(boxes=[])])
        )
        assert result.data == []
        assert result.image is not None

    def test_image_is_plot_output(self, sample_bgr_image, fake_yolo):
        plotted = np.full((6, 6, 3), 9, np.uint8)
        fake_yolo._results[0]._plotted = plotted
        result = detect_objects(image=sample_bgr_image, model=fake_yolo)
        assert np.array_equal(result.image, plotted)

    def test_meta_custom_model(self, sample_bgr_image, fake_yolo):
        result = detect_objects(image=sample_bgr_image, model=fake_yolo)
        assert result.meta["operation"] == "detect_objects"
        assert result.meta["model"] == "custom"
        assert result.meta["source"] is sample_bgr_image

    def test_model_receives_image_array(self, sample_bgr_array, sample_bgr_image, fake_yolo):
        detect_objects(image=sample_bgr_image, model=fake_yolo)
        assert len(fake_yolo.calls) == 1
        assert np.array_equal(fake_yolo.calls[0], sample_bgr_array)

    def test_show_result_invoked(self, sample_bgr_image):
        yolo_result = FakeYOLOResult(boxes=[])
        model = FakeYOLO(results=[yolo_result])
        detect_objects(image=sample_bgr_image, model=model, show_result=True)
        assert yolo_result.show_called is True

    def test_show_result_not_invoked_by_default(self, sample_bgr_image):
        yolo_result = FakeYOLOResult(boxes=[])
        model = FakeYOLO(results=[yolo_result])
        detect_objects(image=sample_bgr_image, model=model)
        assert yolo_result.show_called is False


class TestDetectObjectsModelSelection:
    @pytest.mark.parametrize(
        "level,expected_model",
        [
            (1, "yolo11n.pt"),
            (2, "yolo11s.pt"),
            (3, "yolo11m.pt"),
            (4, "yolo11l.pt"),
            (5, "yolo11x.pt"),
        ],
    )
    def test_accuracy_level_maps_to_model(
        self, monkeypatch, sample_bgr_image, level, expected_model
    ):
        created = {}

        class RecordingYOLO:
            def __init__(self, model):
                created["model"] = model
                self._result = FakeYOLOResult(boxes=[])

            def __call__(self, image):
                return [self._result]

        monkeypatch.setattr(
            "ImagePRO.object_analysis.object_detection.YOLO", RecordingYOLO
        )
        result = detect_objects(image=sample_bgr_image, accuracy_level=level)
        assert created["model"] == expected_model
        assert result.meta["model"] == expected_model

    @pytest.mark.parametrize("level", [0, 6, -1, 99])
    def test_invalid_accuracy_level_raises(self, sample_bgr_image, level):
        with pytest.raises(ValueError):
            detect_objects(image=sample_bgr_image, accuracy_level=level)


class TestDetectObjectsValidation:
    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            detect_objects(image=np.zeros((5, 5, 3)))
