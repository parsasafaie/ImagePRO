"""Unit tests for body pose estimation (MediaPipe Pose faked)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ImagePRO.human_analysis.body_analysis.body_pose_estimation import detect_body_pose

from fakes import FakePose, make_landmarks


class TestDetectBodyPoseNoDetection:
    def test_returns_empty_data_and_annotated_copy(self, sample_bgr_array, sample_bgr_image):
        pose = FakePose()  # pose_landmarks None
        result = detect_body_pose(image=sample_bgr_image, pose_obj=pose)
        assert result.data == []
        assert np.array_equal(result.image, sample_bgr_array)
        assert result.image is not sample_bgr_image._data

    def test_detector_receives_rgb_image(self, sample_bgr_image):
        pose = FakePose()
        detect_body_pose(image=sample_bgr_image, pose_obj=pose)
        assert len(pose.processed_images) == 1
        assert pose.processed_images[0].shape == sample_bgr_image._data.shape


class TestDetectBodyPoseWithDetection:
    def test_all_landmarks_reported(self, sample_bgr_image):
        faces = make_landmarks([[(0, 0.1, 0.2, 0.3), (32, 0.4, 0.5, 0.6)]])
        pose = FakePose(detection_result=SimpleNamespace(
            pose_landmarks=faces[0]
        ))
        result = detect_body_pose(image=sample_bgr_image, pose_obj=pose)
        assert len(result.data) == 33
        assert result.data[0] == [0, 0.1, 0.2, 0.3]
        assert result.data[32] == [32, 0.4, 0.5, 0.6]

    def test_selected_landmarks_only(self, sample_bgr_image):
        faces = make_landmarks([[(5, 0.5, 0.5, 0.5), (17, 0.1, 0.2, 0.3)]])
        pose = FakePose(detection_result=SimpleNamespace(pose_landmarks=faces[0]))
        result = detect_body_pose(
            image=sample_bgr_image, landmarks_idx=[5, 17], pose_obj=pose
        )
        assert result.data == [[5, 0.5, 0.5, 0.5], [17, 0.1, 0.2, 0.3]]

    def test_meta_contents(self, sample_bgr_image):
        pose = FakePose()
        result = detect_body_pose(
            image=sample_bgr_image, min_confidence=0.5, landmarks_idx=[3], pose_obj=pose
        )
        assert result.meta["operation"] == "detect_body_pose"
        assert result.meta["min_confidence"] == 0.5
        assert result.meta["landmarks_idx"] == [3]
        assert result.meta["source"] is sample_bgr_image

    def test_input_not_modified(self, sample_bgr_array, sample_bgr_image):
        before = sample_bgr_array.copy()
        faces = make_landmarks([[(5, 0.5, 0.5, 0.5)]])
        pose = FakePose(detection_result=SimpleNamespace(pose_landmarks=faces[0]))
        detect_body_pose(image=sample_bgr_image, landmarks_idx=[5], pose_obj=pose)
        assert np.array_equal(sample_bgr_image._data, before)


class TestDetectBodyPoseValidation:
    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            detect_body_pose(image=np.zeros((6, 6, 3)))

    @pytest.mark.parametrize("min_confidence", [-0.1, 1.1, "0.5"])
    def test_invalid_confidence_raises(self, sample_bgr_image, min_confidence):
        with pytest.raises(ValueError):
            detect_body_pose(image=sample_bgr_image, min_confidence=min_confidence)

    @pytest.mark.parametrize("landmarks_idx", ["0", [0, "1"], [1.5], (0, 1)])
    def test_invalid_landmarks_idx_raises(self, sample_bgr_image, landmarks_idx):
        with pytest.raises(TypeError):
            detect_body_pose(image=sample_bgr_image, landmarks_idx=landmarks_idx)
