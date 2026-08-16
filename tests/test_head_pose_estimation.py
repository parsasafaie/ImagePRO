"""Unit tests for head pose estimation (analyze_face_mesh mocked)."""

from __future__ import annotations

import pytest

from ImagePRO.human_analysis.face_analysis import head_pose_estimation
from ImagePRO.human_analysis.face_analysis.head_pose_estimation import estimate_head_pose
from ImagePRO.utils.result import Result


def patch_mesh(monkeypatch, faces_data):
    monkeypatch.setattr(
        head_pose_estimation,
        "analyze_face_mesh",
        lambda **kwargs: Result(image=None, data=faces_data, meta={}),
    )


def pose_rows(left_x, nasion_x, right_x, nasion_y, nose_y, chin_y):
    # Landmark rows for indices 1 (nose), 152 (chin), 33, 263 (eyes),
    # 168 (nasion). Format: [face_id, idx, x, y, z].
    return [
        [0, 1, 0.5, nose_y, 0.0],
        [0, 152, 0.5, chin_y, 0.0],
        [0, 33, left_x, 0.5, 0.0],
        [0, 263, right_x, 0.5, 0.0],
        [0, 168, nasion_x, nasion_y, 0.0],
    ]


class TestEstimateHeadPose:
    def test_frontal_face_gives_zero_yaw(self, sample_bgr_image, monkeypatch):
        rows = pose_rows(
            left_x=0.3, nasion_x=0.5, right_x=0.7,
            nasion_y=0.4, nose_y=0.5, chin_y=0.9,
        )
        patch_mesh(monkeypatch, [rows])
        result = estimate_head_pose(image=sample_bgr_image, face_mesh_obj=object())
        assert result.data[0][0] == 0  # face id
        assert result.data[0][1] == pytest.approx(0.0)  # yaw
        # pitch = 100 * ((0.9 - 0.5) - (0.5 - 0.4)) = 30
        assert result.data[0][2] == pytest.approx(30.0)

    def test_turned_face_gives_nonzero_yaw(self, sample_bgr_image, monkeypatch):
        rows = pose_rows(
            left_x=0.2, nasion_x=0.35, right_x=0.8,
            nasion_y=0.4, nose_y=0.5, chin_y=0.9,
        )
        patch_mesh(monkeypatch, [rows])
        result = estimate_head_pose(image=sample_bgr_image, face_mesh_obj=object())
        # yaw = 100 * ((0.8-0.35) - (0.35-0.2)) = 30
        assert result.data[0][1] == pytest.approx(30.0)

    def test_no_face_returns_none_with_error(self, sample_bgr_image, monkeypatch):
        patch_mesh(monkeypatch, [])
        result = estimate_head_pose(image=sample_bgr_image, face_mesh_obj=object())
        assert result.data is None
        assert result.meta["error"] == "No face landmarks detected"

    def test_missing_landmark_returns_none_with_error(self, sample_bgr_image, monkeypatch):
        patch_mesh(monkeypatch, [[0, 1, 0.5, 0.5, 0.0]])  # others missing
        result = estimate_head_pose(image=sample_bgr_image, face_mesh_obj=object())
        assert result.data is None
        assert result.meta["error"] == "Missing required landmarks"

    def test_meta_contents(self, sample_bgr_image, monkeypatch):
        rows = pose_rows(0.3, 0.5, 0.7, 0.4, 0.5, 0.9)
        patch_mesh(monkeypatch, [rows])
        result = estimate_head_pose(
            image=sample_bgr_image, max_faces=5, min_confidence=0.6, face_mesh_obj=object()
        )
        assert result.meta["operation"] == "estimate_head_pose"
        assert result.meta["max_faces"] == 5
        assert result.meta["min_confidence"] == 0.6
        assert result.image is None


class TestEstimateHeadPoseValidation:
    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            estimate_head_pose(image=None)

    @pytest.mark.parametrize("max_faces", [0, -1, 2.5, "1"])
    def test_invalid_max_faces_raises(self, sample_bgr_image, max_faces):
        with pytest.raises(ValueError):
            estimate_head_pose(image=sample_bgr_image, max_faces=max_faces)

    @pytest.mark.parametrize("min_confidence", [-1.0, 1.5, "0.7"])
    def test_invalid_confidence_raises(self, sample_bgr_image, min_confidence):
        with pytest.raises(ValueError):
            estimate_head_pose(image=sample_bgr_image, min_confidence=min_confidence)
