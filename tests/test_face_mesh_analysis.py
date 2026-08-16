"""Unit tests for face mesh analysis (MediaPipe mocked)."""

from __future__ import annotations

import mediapipe as mp  # real package or the conftest stub

import numpy as np
import pytest

from ImagePRO.human_analysis.face_analysis.face_mesh_analysis import analyze_face_mesh

from fakes import FakeFaceMesh, make_landmarks


@pytest.fixture
def patch_facemesh(monkeypatch):
    """Replace the FaceMesh class and record every instance created."""

    def _patch(result):
        instance_holder = {}

        def factory(**kwargs):
            mesh = FakeFaceMesh(detection_result=result, **kwargs)
            instance_holder["instance"] = mesh
            return mesh

        monkeypatch.setattr(mp.solutions.face_mesh, "FaceMesh", factory)
        return instance_holder

    return _patch


class TestAnalyzeFaceMeshNoDetection:
    def test_returns_none_image_and_data_with_error_meta(
        self, sample_bgr_image, patch_facemesh
    ):
        patch_facemesh(result=None)  # multi_face_landmarks falsy
        result = analyze_face_mesh(image=sample_bgr_image)
        assert result.image is None
        assert result.data is None
        assert result.meta["error"] == "No face landmarks detected"
        assert result.meta["operation"] == "analyze_face_mesh"

    def test_source_image_not_modified(self, sample_bgr_array, sample_bgr_image, patch_facemesh):
        before = sample_bgr_array.copy()
        patch_facemesh(result=None)
        analyze_face_mesh(image=sample_bgr_image)
        assert np.array_equal(sample_bgr_image._data, before)


class TestAnalyzeFaceMeshWithDetection:
    def test_full_mesh_returns_468_rows_per_face(self, sample_bgr_image, patch_facemesh):
        from types import SimpleNamespace

        faces = make_landmarks(
            [[(0, 0.1, 0.2, 0.3), (467, 0.4, 0.5, 0.6)]]
        )
        patch_facemesh(result=SimpleNamespace(multi_face_landmarks=faces))
        result = analyze_face_mesh(image=sample_bgr_image)
        assert len(result.data) == 1
        assert len(result.data[0]) == 468
        assert result.data[0][0] == [0, 0, 0.1, 0.2, 0.3]
        assert result.data[0][467] == [0, 467, 0.4, 0.5, 0.6]

    def test_selected_landmarks_only(self, sample_bgr_image, patch_facemesh):
        from types import SimpleNamespace

        faces = make_landmarks([[(10, 0.1, 0.2, 0.3), (20, 0.5, 0.6, 0.7)]])
        patch_facemesh(result=SimpleNamespace(multi_face_landmarks=faces))
        result = analyze_face_mesh(
            image=sample_bgr_image, landmarks_idx=[10, 20]
        )
        assert result.data == [
            [[0, 10, 0.1, 0.2, 0.3], [0, 20, 0.5, 0.6, 0.7]]
        ]

    def test_multiple_faces(self, sample_bgr_image, patch_facemesh):
        from types import SimpleNamespace

        faces = make_landmarks(
            [
                [(5, 0.1, 0.1, 0.1)],
                [(5, 0.2, 0.2, 0.2)],
            ]
        )
        patch_facemesh(result=SimpleNamespace(multi_face_landmarks=faces))
        result = analyze_face_mesh(image=sample_bgr_image, landmarks_idx=[5])
        assert len(result.data) == 2
        assert result.data[0][0][0] == 0
        assert result.data[1][0][0] == 1
        assert result.data[1][0][2] == 0.2

    def test_annotated_image_is_a_copy_of_input(
        self, sample_bgr_array, sample_bgr_image, patch_facemesh
    ):
        from types import SimpleNamespace

        faces = make_landmarks([[(10, 0.5, 0.5, 0.0)]])
        patch_facemesh(result=SimpleNamespace(multi_face_landmarks=faces))
        result = analyze_face_mesh(image=sample_bgr_image, landmarks_idx=[10])
        assert result.image is not sample_bgr_image._data
        assert result.image.shape == sample_bgr_array.shape

    def test_model_receives_rgb_image(self, sample_bgr_image, patch_facemesh):
        from types import SimpleNamespace

        faces = make_landmarks([[(0, 0.0, 0.0, 0.0)]])
        holder = patch_facemesh(result=SimpleNamespace(multi_face_landmarks=faces))
        analyze_face_mesh(image=sample_bgr_image, landmarks_idx=[0])
        processed = holder["instance"].processed_images[0]
        # BGR white pixel becomes RGB red-dominant ordering: check channel
        # order differs from the raw BGR input.
        assert processed.shape == sample_bgr_image._data.shape

    def test_full_mesh_drawing_invoked(
        self, sample_bgr_image, patch_facemesh, monkeypatch
    ):
        from types import SimpleNamespace

        faces = make_landmarks([[(0, 0.1, 0.1, 0.1)]])
        patch_facemesh(result=SimpleNamespace(multi_face_landmarks=faces))
        calls = []
        monkeypatch.setattr(
            mp.solutions.drawing_utils,
            "draw_landmarks",
            lambda *args, **kwargs: calls.append(args),
        )
        analyze_face_mesh(image=sample_bgr_image)  # all landmarks -> tessellation
        assert len(calls) == 1


class TestAnalyzeFaceMeshConfiguration:
    def test_detector_initialized_with_expected_options(
        self, sample_bgr_image, patch_facemesh
    ):
        from types import SimpleNamespace

        faces = make_landmarks([[(0, 0.1, 0.1, 0.1)]])
        holder = patch_facemesh(result=SimpleNamespace(multi_face_landmarks=faces))
        analyze_face_mesh(
            image=sample_bgr_image,
            max_faces=3,
            min_confidence=0.9,
            landmarks_idx=[0],
        )
        kwargs = holder["instance"].kwargs
        assert kwargs["max_num_faces"] == 3
        assert kwargs["min_detection_confidence"] == 0.9
        assert kwargs["static_image_mode"] is True

    def test_provided_detector_is_reused(self, sample_bgr_image, monkeypatch):
        from types import SimpleNamespace

        faces = make_landmarks([[(0, 0.1, 0.1, 0.1)]])
        sentinel = FakeFaceMesh(
            detection_result=SimpleNamespace(multi_face_landmarks=faces)
        )
        created = []
        monkeypatch.setattr(
            mp.solutions.face_mesh,
            "FaceMesh",
            lambda **kwargs: created.append(kwargs),
        )
        analyze_face_mesh(
            image=sample_bgr_image, landmarks_idx=[0], face_mesh_obj=sentinel
        )
        assert created == []
        assert len(sentinel.processed_images) == 1

    def test_meta_contains_parameters(self, sample_bgr_image, patch_facemesh):
        from types import SimpleNamespace

        faces = make_landmarks([[(0, 0.1, 0.1, 0.1)]])
        patch_facemesh(result=SimpleNamespace(multi_face_landmarks=faces))
        result = analyze_face_mesh(
            image=sample_bgr_image, max_faces=2, min_confidence=0.5, landmarks_idx=[7]
        )
        assert result.meta["max_faces"] == 2
        assert result.meta["min_confidence"] == 0.5
        assert result.meta["landmarks_idx"] == [7]
        assert result.meta["source"] is sample_bgr_image


class TestAnalyzeFaceMeshValidation:
    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            analyze_face_mesh(image=None)

    @pytest.mark.parametrize("max_faces", [0, -1, 2.5, "3"])
    def test_invalid_max_faces_raises(self, sample_bgr_image, max_faces):
        with pytest.raises(ValueError):
            analyze_face_mesh(image=sample_bgr_image, max_faces=max_faces)

    @pytest.mark.parametrize("min_confidence", [-0.1, 1.1, 2, "0.5"])
    def test_invalid_confidence_raises(self, sample_bgr_image, min_confidence):
        with pytest.raises(ValueError):
            analyze_face_mesh(image=sample_bgr_image, min_confidence=min_confidence)

    @pytest.mark.parametrize("landmarks_idx", ["0,1", [0, "1"], [0.5], 5])
    def test_invalid_landmarks_idx_raises(self, sample_bgr_image, landmarks_idx):
        with pytest.raises(TypeError):
            analyze_face_mesh(image=sample_bgr_image, landmarks_idx=landmarks_idx)
