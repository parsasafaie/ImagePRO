"""Unit tests for ImagePRO.pre_processing.blur."""

from __future__ import annotations

import numpy as np
import pytest

from ImagePRO.pre_processing.blur import (
    apply_average_blur,
    apply_bilateral_blur,
    apply_gaussian_blur,
    apply_median_blur,
)
from ImagePRO.utils.image import Image


@pytest.fixture
def constant_image():
    return Image.from_array(np.full((20, 30, 3), 100, np.uint8))


class TestAverageBlur:
    def test_constant_image_unchanged(self, constant_image):
        result = apply_average_blur(image=constant_image)
        assert np.array_equal(result.image, constant_image._data)

    def test_shape_and_dtype_preserved(self, sample_bgr_image):
        result = apply_average_blur(image=sample_bgr_image)
        assert result.image.shape == sample_bgr_image.shape
        assert result.image.dtype == np.uint8

    def test_output_is_a_new_array(self, sample_bgr_image):
        result = apply_average_blur(image=sample_bgr_image)
        assert result.image is not sample_bgr_image._data

    def test_meta_contents(self, sample_bgr_image):
        result = apply_average_blur(image=sample_bgr_image, kernel_size=(3, 3))
        assert result.data is None
        assert result.meta["operation"] == "apply_average_blur"
        assert result.meta["kernel_size"] == (3, 3)
        assert result.meta["source"] is sample_bgr_image

    def test_non_image_raises(self, sample_bgr_array):
        with pytest.raises(TypeError):
            apply_average_blur(image=sample_bgr_array)

    @pytest.mark.parametrize(
        "kernel", [(0, 5), (5, 0), (-3, 3), (3,), (3, 3, 3), [3, 3], (2.5, 3), ("a", 3)]
    )
    def test_invalid_kernel_raises(self, sample_bgr_image, kernel):
        with pytest.raises(ValueError):
            apply_average_blur(image=sample_bgr_image, kernel_size=kernel)


class TestGaussianBlur:
    def test_constant_image_unchanged(self, constant_image):
        result = apply_gaussian_blur(image=constant_image)
        assert np.array_equal(result.image, constant_image._data)

    def test_shape_preserved(self, sample_bgr_image):
        result = apply_gaussian_blur(image=sample_bgr_image, kernel_size=(7, 7))
        assert result.image.shape == sample_bgr_image.shape

    def test_meta_contents(self, sample_bgr_image):
        result = apply_gaussian_blur(image=sample_bgr_image)
        assert result.meta["operation"] == "apply_gaussian_blur"
        assert result.meta["kernel_size"] == (5, 5)

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            apply_gaussian_blur(image=None)

    @pytest.mark.parametrize(
        "kernel", [(4, 5), (5, 4), (0, 5), (-5, 5), (3,), (3, 3, 3), (2.5, 5)]
    )
    def test_even_or_invalid_kernel_raises(self, sample_bgr_image, kernel):
        with pytest.raises(ValueError):
            apply_gaussian_blur(image=sample_bgr_image, kernel_size=kernel)


class TestMedianBlur:
    @pytest.fixture
    def noisy_image(self):
        clean = np.full((15, 15, 3), 50, np.uint8)
        noisy = clean.copy()
        noisy[7, 7] = 255  # single salt-noise pixel
        return Image.from_array(noisy), clean

    def test_removes_salt_noise(self, noisy_image):
        noisy, clean = noisy_image
        result = apply_median_blur(image=noisy, filter_size=3)
        assert np.array_equal(result.image, clean)

    def test_meta_contents(self, sample_bgr_image):
        result = apply_median_blur(image=sample_bgr_image)
        assert result.meta["operation"] == "apply_median_blur"
        assert result.meta["filter_size"] == 5

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            apply_median_blur(image="image")

    @pytest.mark.parametrize("size", [0, 1, 2, 4, -3, 3.5, "5"])
    def test_invalid_filter_size_raises(self, sample_bgr_image, size):
        with pytest.raises(ValueError):
            apply_median_blur(image=sample_bgr_image, filter_size=size)


class TestBilateralBlur:
    def test_constant_image_unchanged(self, constant_image):
        result = apply_bilateral_blur(image=constant_image)
        assert np.array_equal(result.image, constant_image._data)

    def test_shape_preserved(self, sample_bgr_image):
        result = apply_bilateral_blur(image=sample_bgr_image)
        assert result.image.shape == sample_bgr_image.shape

    def test_meta_contents(self, sample_bgr_image):
        result = apply_bilateral_blur(
            image=sample_bgr_image,
            filter_size=7,
            sigma_color=40,
            sigma_space=60,
        )
        assert result.meta["operation"] == "apply_bilateral_blur"
        assert result.meta["filter_size"] == 7
        assert result.meta["sigma_color"] == 40
        assert result.meta["sigma_space"] == 60

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            apply_bilateral_blur(image=123)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"filter_size": 0},
            {"filter_size": -1},
            {"filter_size": 2.5},
            {"sigma_color": 0},
            {"sigma_color": -10},
            {"sigma_space": 0},
            {"sigma_space": "big"},
        ],
    )
    def test_invalid_parameters_raise(self, sample_bgr_image, kwargs):
        with pytest.raises(ValueError):
            apply_bilateral_blur(image=sample_bgr_image, **kwargs)
