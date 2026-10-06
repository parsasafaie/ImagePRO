"""Unit tests for ImagePRO.utils.image.Image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ImagePRO.utils.image import Image


class TestFromArray:
    def test_wraps_array_content(self, sample_bgr_array):
        image = Image.from_array(sample_bgr_array)
        assert np.array_equal(image._data, sample_bgr_array)

    def test_defaults(self, sample_bgr_array):
        image = Image.from_array(sample_bgr_array)
        assert image.colorspace == "BGR"
        assert image.source_type == "array"
        assert image.path is None

    def test_custom_colorspace(self, sample_bgr_array):
        image = Image.from_array(sample_bgr_array, colorspace="RGB")
        assert image.colorspace == "RGB"

    @pytest.mark.parametrize("colorspace", ["rgb", "HSV", "", None, 3])
    def test_invalid_colorspace_raises(self, sample_bgr_array, colorspace):
        with pytest.raises(ValueError):
            Image.from_array(sample_bgr_array, colorspace=colorspace)

    @pytest.mark.parametrize("bad", [None, "path.jpg", 42, [1, 2, 3]])
    def test_non_ndarray_raises(self, bad):
        with pytest.raises(TypeError):
            Image.from_array(bad)


class TestFromPath:
    def test_loads_file(self, image_file, sample_bgr_array):
        image = Image.from_path(image_file)
        assert image._data.shape == sample_bgr_array.shape
        assert image._data.dtype == np.uint8
        assert image.path == Path(image_file)
        assert image.source_type == "path"
        assert image.colorspace == "BGR"

    def test_accepts_string_path(self, image_file):
        image = Image.from_path(str(image_file))
        assert image.source_type == "path"

    def test_roundtrip_preserves_content(self, image_file, sample_bgr_array):
        image = Image.from_path(image_file)
        assert np.array_equal(image._data, sample_bgr_array)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Image.from_path(tmp_path / "does_not_exist.png")

    @pytest.mark.parametrize("bad", [None, 42, b"bytes.png"])
    def test_invalid_path_type_raises(self, bad):
        with pytest.raises(TypeError):
            Image.from_path(bad)

    def test_invalid_colorspace_raises(self, image_file):
        with pytest.raises(ValueError):
            Image.from_path(image_file, colorspace="YUV")

    def test_gray_load_produces_2d_array(self, image_file, sample_bgr_array):
        # Regression: from_path ignored the colorspace flag and always
        # returned BGR data, so a "GRAY"-tagged image carried 3 channels and
        # broke every grayscale-path function downstream.
        import cv2

        image = Image.from_path(image_file, colorspace="GRAY")
        assert image._data.ndim == 2
        assert image.colorspace == "GRAY"
        expected = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(image._data, expected)

    def test_rgb_load_converts_channels(self, image_file, sample_bgr_array):
        image = Image.from_path(image_file, colorspace="RGB")
        assert image._data.shape == sample_bgr_array.shape
        assert np.array_equal(image._data, sample_bgr_array[..., ::-1])


class TestProperties:
    def test_shape_color(self, sample_bgr_image):
        assert sample_bgr_image.shape == (24, 32, 3)

    def test_shape_gray(self, sample_gray_image):
        assert sample_gray_image.shape == (12, 10)

    def test_dtype(self, sample_bgr_image):
        assert sample_bgr_image.dtype == np.uint8


class TestImmutability:
    def test_attribute_assignment_raises(self, sample_bgr_image):
        from dataclasses import FrozenInstanceError

        # Regression: the dataclass was not frozen, so callers could change
        # the colorspace tag without touching the data.
        with pytest.raises(FrozenInstanceError):
            sample_bgr_image.colorspace = "RGB"
        with pytest.raises(FrozenInstanceError):
            sample_bgr_image.source_type = "path"

    def test_dataclass_declared_frozen(self):
        assert Image.__dataclass_params__.frozen is True
