"""Unit tests for ImagePRO.utils.result.Result and flatten_rows."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from ImagePRO.utils.result import Result, flatten_rows


@pytest.fixture
def result_with_image(sample_bgr_array):
    return Result(image=sample_bgr_array.copy())


class TestSaveAsImgSingle:
    def test_saves_file_and_returns_self(self, result_with_image, tmp_path):
        out = tmp_path / "out.png"
        returned = result_with_image.save_as_img(out)
        assert returned is result_with_image
        assert out.exists()

    def test_saved_content_roundtrips(self, result_with_image, tmp_path):
        import cv2

        out = tmp_path / "out.png"
        result_with_image.save_as_img(out)
        loaded = cv2.imread(str(out))
        assert np.array_equal(loaded, result_with_image.image)

    def test_creates_parent_directories(self, result_with_image, tmp_path):
        out = tmp_path / "nested" / "deeper" / "out.png"
        result_with_image.save_as_img(out)
        assert out.exists()

    def test_no_image_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Result().save_as_img(tmp_path / "out.png")

    @pytest.mark.parametrize("bad", [None, 42, ["not_a_path"]])
    def test_invalid_path_type_raises(self, result_with_image, bad):
        with pytest.raises(TypeError):
            result_with_image.save_as_img(bad)

    @pytest.mark.parametrize("bad_image", ["string", 42, 3.14])
    def test_invalid_image_type_raises(self, bad_image, tmp_path):
        with pytest.raises(TypeError):
            Result(image=bad_image).save_as_img(tmp_path / "out.png")

    def test_invalid_list_contents_raises(self, tmp_path):
        with pytest.raises(TypeError):
            Result(image=[np.zeros((2, 2, 3), np.uint8), "not_array"]).save_as_img(
                tmp_path / "out.png"
            )


class TestSaveAsImgList:
    def test_saves_list_with_suffixed_names(self, tmp_path):
        images = [
            np.zeros((4, 4, 3), np.uint8),
            np.full((4, 4, 3), 7, np.uint8),
            np.full((4, 4, 3), 9, np.uint8),
        ]
        Result(image=images).save_as_img(tmp_path / "out.jpg")
        assert (tmp_path / "out.jpg").exists()
        assert (tmp_path / "out_1.jpg").exists()
        assert (tmp_path / "out_2.jpg").exists()

    def test_non_jpg_suffix_uses_index_before_extension(self, tmp_path):
        images = [np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8)]
        Result(image=images).save_as_img(tmp_path / "out.png")
        assert (tmp_path / "out.png").exists()
        assert (tmp_path / "out_1.png").exists()

    def test_list_contents_saved(self, tmp_path):
        import cv2

        images = [np.full((4, 4, 3), 42, np.uint8), np.full((4, 4, 3), 84, np.uint8)]
        Result(image=images).save_as_img(tmp_path / "out.png")
        assert cv2.imread(str(tmp_path / "out.png")).flat[0] == 42
        assert cv2.imread(str(tmp_path / "out_1.png")).flat[0] == 84


class TestSaveAsCsv:
    def test_writes_result_data(self, tmp_path):
        out = tmp_path / "data.csv"
        Result(data=[[1, 2], [3, 4]]).save_as_csv(out)
        with out.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows == [["1", "2"], ["3", "4"]]

    def test_explicit_rows_override_data(self, tmp_path):
        out = tmp_path / "data.csv"
        Result(data=[[1, 2]]).save_as_csv(out, rows=[[9, 9]])
        with out.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows == [["9", "9"]]

    def test_non_list_payload_writes_single_row(self, tmp_path):
        out = tmp_path / "data.csv"
        Result(data="hello").save_as_csv(out)
        with out.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows == [["hello"]]

    def test_no_data_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Result().save_as_csv(tmp_path / "data.csv")

    @pytest.mark.parametrize("bad", [None, 42, ["x"]])
    def test_invalid_path_type_raises(self, bad):
        with pytest.raises(TypeError):
            Result(data=[[1]]).save_as_csv(bad)

    def test_write_failure_raises_ioerror(self, tmp_path):
        with pytest.raises(IOError):
            Result(data=[[1]]).save_as_csv(tmp_path)  # a directory

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "data.csv"
        Result(data=[[1]]).save_as_csv(out)
        assert out.exists()

    def test_returns_self_for_chaining(self, tmp_path):
        result = Result(data=[[1]])
        assert result.save_as_csv(tmp_path / "a.csv") is result


class TestFlattenRows:
    def test_flat_rows_unchanged(self):
        assert flatten_rows([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    def test_nested_list_of_rows_recurses(self):
        assert flatten_rows([[[1, 2], [3, 4]]]) == [[1, 2], [3, 4]]

    def test_scalars_become_single_cell_rows(self):
        assert flatten_rows([1, "a"]) == [[1], ["a"]]

    def test_mixed_row_kept_as_is(self):
        # Actual behavior: a row with a nested value inside is not
        # flattened further (only rows whose items are all lists/tuples
        # recurse). The docstring example suggesting otherwise is
        # inaccurate; this pins the real behavior.
        assert flatten_rows([[3, [4, 5]]]) == [[3, [4, 5]]]

    def test_tuples_supported(self):
        assert flatten_rows([(1, 2)]) == [[1, 2]]

    def test_empty_row_becomes_empty_list_row(self):
        assert flatten_rows([[]]) == [[]]

    def test_landmark_style_rows(self):
        rows = [[0, 10, 0.5, 0.25, -0.1], [0, 11, 0.6, 0.35, -0.2]]
        assert flatten_rows(rows) == rows


class TestResultDefaults:
    def test_default_fields(self):
        result = Result()
        assert result.image is None
        assert result.data is None
        assert result.meta == {}

    def test_meta_is_independent_per_instance(self):
        first, second = Result(), Result()
        first.meta["key"] = "value"
        assert "key" not in second.meta
