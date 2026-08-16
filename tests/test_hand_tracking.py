"""Unit tests for hand tracking (MediaPipe Hands faked)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ImagePRO.human_analysis.body_analysis.hand_tracking import detect_hands

from fakes import FakeHands, make_landmarks


class TestDetectHandsNoDetection:
    def test_returns_empty_data_and_annotated_copy(self, sample_bgr_array, sample_bgr_image):
        hands = FakeHands()  # multi_hand_landmarks None
        result = detect_hands(image=sample_bgr_image, hands_obj=hands)
        assert result.data == []
        assert np.array_equal(result.image, sample_bgr_array)

    def test_detector_receives_rgb_image(self, sample_bgr_image):
        hands = FakeHands()
        detect_hands(image=sample_bgr_image, hands_obj=hands)
        assert len(hands.processed_images) == 1


class TestDetectHandsWithDetection:
    def test_all_landmarks_for_single_hand(self, sample_bgr_image):
        faces = make_landmarks([[(0, 0.1, 0.2, 0.3), (20, 0.6, 0.7, 0.8)]])
        hands = FakeHands(detection_result=SimpleNamespace(multi_hand_landmarks=faces))
        result = detect_hands(image=sample_bgr_image, hands_obj=hands)
        assert len(result.data) == 21
        assert result.data[0] == [0, 0, 0.1, 0.2, 0.3]
        assert result.data[20] == [0, 20, 0.6, 0.7, 0.8]

    def test_two_hands_have_distinct_ids(self, sample_bgr_image):
        faces = make_landmarks(
            [
                [(4, 0.1, 0.1, 0.1)],
                [(8, 0.2, 0.2, 0.2)],
            ]
        )
        hands = FakeHands(detection_result=SimpleNamespace(multi_hand_landmarks=faces))
        result = detect_hands(
            image=sample_bgr_image, landmarks_idx=[4, 8], hands_obj=hands
        )
        # Each hand reports every requested index; unspecified landmarks
        # fall back to the detector default (0.0).
        assert result.data == [
            [0, 4, 0.1, 0.1, 0.1],
            [0, 8, 0.0, 0.0, 0.0],
            [1, 4, 0.0, 0.0, 0.0],
            [1, 8, 0.2, 0.2, 0.2],
        ]

    def test_selected_landmarks_only(self, sample_bgr_image):
        faces = make_landmarks([[(0, 0.5, 0.5, 0.5)]])
        hands = FakeHands(detection_result=SimpleNamespace(multi_hand_landmarks=faces))
        result = detect_hands(image=sample_bgr_image, landmarks_idx=[0], hands_obj=hands)
        assert result.data == [[0, 0, 0.5, 0.5, 0.5]]

    def test_meta_contents(self, sample_bgr_image):
        hands = FakeHands()
        result = detect_hands(
            image=sample_bgr_image,
            max_hands=1,
            min_confidence=0.8,
            hands_obj=hands,
        )
        assert result.meta["operation"] == "detect_hands"
        assert result.meta["max_hands"] == 1
        assert result.meta["min_confidence"] == 0.8
        assert result.meta["source"] is sample_bgr_image

    def test_input_not_modified(self, sample_bgr_array, sample_bgr_image):
        before = sample_bgr_array.copy()
        faces = make_landmarks([[(0, 0.5, 0.5, 0.5)]])
        hands = FakeHands(detection_result=SimpleNamespace(multi_hand_landmarks=faces))
        detect_hands(image=sample_bgr_image, landmarks_idx=[0], hands_obj=hands)
        assert np.array_equal(sample_bgr_image._data, before)


class TestDetectHandsValidation:
    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            detect_hands(image=123)

    @pytest.mark.parametrize("max_hands", [0, -1, 1.5, "2"])
    def test_invalid_max_hands_raises(self, sample_bgr_image, max_hands):
        with pytest.raises(ValueError):
            detect_hands(image=sample_bgr_image, max_hands=max_hands)

    @pytest.mark.parametrize("min_confidence", [-0.2, 1.2, "0.5"])
    def test_invalid_confidence_raises(self, sample_bgr_image, min_confidence):
        with pytest.raises(ValueError):
            detect_hands(image=sample_bgr_image, min_confidence=min_confidence)

    @pytest.mark.parametrize("landmarks_idx", ["0", [0, 1.5], 7])
    def test_invalid_landmarks_idx_raises(self, sample_bgr_image, landmarks_idx):
        with pytest.raises(TypeError):
            detect_hands(image=sample_bgr_image, landmarks_idx=landmarks_idx)
