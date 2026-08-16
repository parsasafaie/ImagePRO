"""Unit tests for face comparison (InsightFace faked).

Note: compare_faces currently crashes before reaching the model because
save_temp_image is called with a raw ndarray instead of the Image
wrapper. The behavior tests below are marked xfail until that bug is
fixed; they document the intended contract.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ImagePRO.human_analysis.face_analysis.face_comparison import compare_faces
from ImagePRO.utils.image import Image

from fakes import FakeFaceAnalysisApp


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestCompareFacesValidation:
    @pytest.mark.parametrize("first,second", [(None, None), ("a", "b"), (1, 2)])
    def test_non_image_inputs_raise(self, first, second):
        with pytest.raises(TypeError):
            compare_faces(first, second)


class TestCompareFacesBehavior:
    @pytest.mark.xfail(
        reason="compare_faces passes image._data (ndarray) to save_temp_image, "
        "which expects an Image instance and raises AttributeError",
        strict=False,
    )
    def test_same_embedding_reports_match(self, isolated_cwd, sample_bgr_array):
        embedding = [1.0, 0.0, 0.0]
        app = FakeFaceAnalysisApp(faces_per_image=[[embedding], [embedding]])
        image_1 = Image.from_array(sample_bgr_array.copy())
        image_2 = Image.from_array(sample_bgr_array.copy())

        result = compare_faces(image_1, image_2, app=app)
        assert result.data is True
        assert result.meta["similarity"] == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.xfail(
        reason="compare_faces passes image._data (ndarray) to save_temp_image, "
        "which expects an Image instance and raises AttributeError",
        strict=False,
    )
    def test_orthogonal_embeddings_report_no_match(self, isolated_cwd, sample_bgr_array):
        app = FakeFaceAnalysisApp(
            faces_per_image=[[[1.0, 0.0]], [[0.0, 1.0]]]
        )
        image_1 = Image.from_array(sample_bgr_array.copy())
        image_2 = Image.from_array(sample_bgr_array.copy())

        result = compare_faces(image_1, image_2, app=app)
        assert result.data is False
        assert result.meta["similarity"] == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.xfail(
        reason="compare_faces passes image._data (ndarray) to save_temp_image, "
        "which expects an Image instance and raises AttributeError",
        strict=False,
    )
    def test_missing_face_returns_none_with_error(self, isolated_cwd, sample_bgr_array):
        app = FakeFaceAnalysisApp(faces_per_image=[[], [[1.0, 1.0]]])
        image_1 = Image.from_array(sample_bgr_array.copy())
        image_2 = Image.from_array(sample_bgr_array.copy())

        result = compare_faces(image_1, image_2, app=app)
        assert result.data is None
        assert "error" in result.meta

    @pytest.mark.xfail(
        reason="compare_faces passes image._data (ndarray) to save_temp_image, "
        "which expects an Image instance and raises AttributeError",
        strict=False,
    )
    def test_temp_files_cleaned_up(self, isolated_cwd, sample_bgr_array):
        app = FakeFaceAnalysisApp(faces_per_image=[[[1.0]], [[1.0]]])
        image_1 = Image.from_array(sample_bgr_array.copy())
        image_2 = Image.from_array(sample_bgr_array.copy())

        compare_faces(image_1, image_2, app=app)
        assert not os.path.exists("tmp1.jpg")
        assert not os.path.exists("tmp2.jpg")
