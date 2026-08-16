"""Unit tests for ImagePRO.pre_processing.contrast."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ImagePRO.pre_processing.contrast import (
    apply_clahe_contrast,
    apply_contrast_stretching,
    apply_histogram_equalization,
)
from ImagePRO.utils.image import Image


@pytest.fixture
def low_contrast_image():
    # 8x8 image using only values 100..115: grayscale conversion yields
    # a narrow intensity band, ideal for equalization/stretching tests.
    values = np.arange(64).reshape(8, 8) % 16 + 100
    array = np.stack([values, values, values], axis=-1).astype(np.uint8)
    return Image.from_array(array)


class TestClaheContrast:
    def test_output_is_single_channel(self, sample_bgr_image):
        result = apply_clahe_contrast(image=sample_bgr_image)
        assert result.image.ndim == 2
        assert result.image.dtype == np.uint8

    def test_output_values_in_valid_range(self, sample_bgr_image):
        result = apply_clahe_contrast(image=sample_bgr_image)
        assert result.image.min() >= 0
        assert result.image.max() <= 255

    def test_meta_contents(self, sample_bgr_image):
        result = apply_clahe_contrast(
            image=sample_bgr_image, clip_limit=3.0, tile_grid_size=(4, 4)
        )
        assert result.data is None
        assert result.meta["operation"] == "apply_clahe_contrast"
        assert result.meta["clip_limit"] == 3.0
        assert result.meta["tile_grid_size"] == (4, 4)

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            apply_clahe_contrast(image=None)

    @pytest.mark.parametrize("clip_limit", [0, -1.5, "2"])
    def test_invalid_clip_limit_raises(self, sample_bgr_image, clip_limit):
        with pytest.raises(ValueError):
            apply_clahe_contrast(image=sample_bgr_image, clip_limit=clip_limit)

    @pytest.mark.parametrize(
        "grid", [(0, 8), (8, 0), (-8, 8), (8,), (8, 8, 8), [8, 8], (8.0, 8)]
    )
    def test_invalid_tile_grid_raises(self, sample_bgr_image, grid):
        with pytest.raises(TypeError):
            apply_clahe_contrast(image=sample_bgr_image, tile_grid_size=grid)


class TestHistogramEqualization:
    def test_output_is_single_channel(self, sample_bgr_image):
        result = apply_histogram_equalization(image=sample_bgr_image)
        assert result.image.ndim == 2
        assert result.image.dtype == np.uint8

    def test_low_contrast_image_stretched_to_full_range(self, low_contrast_image):
        result = apply_histogram_equalization(image=low_contrast_image)
        assert result.image.min() == 0
        assert result.image.max() == 255

    def test_meta_contents(self, sample_bgr_image):
        result = apply_histogram_equalization(image=sample_bgr_image)
        assert result.data is None
        assert result.meta["operation"] == "apply_histogram_equalization"

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            apply_histogram_equalization(image=42)


class TestContrastStretching:
    def test_identity_transform_returns_grayscale_version(
        self, sample_bgr_array, sample_bgr_image
    ):
        result = apply_contrast_stretching(
            image=sample_bgr_image, alpha=1.0, beta=0
        )
        expected = cv2.cvtColor(sample_bgr_array, cv2.COLOR_BGR2GRAY)
        assert np.array_equal(result.image, expected)

    def test_output_is_single_channel(self, sample_bgr_image):
        result = apply_contrast_stretching(image=sample_bgr_image)
        assert result.image.ndim == 2

    def test_meta_contents(self, sample_bgr_image):
        result = apply_contrast_stretching(image=sample_bgr_image, alpha=1.5, beta=10)
        assert result.meta["operation"] == "apply_contrast_stretching"
        assert result.meta["alpha"] == 1.5
        assert result.meta["beta"] == 10

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            apply_contrast_stretching(image="img")

    @pytest.mark.parametrize("alpha", [-0.5, "2"])
    def test_negative_alpha_raises(self, sample_bgr_image, alpha):
        with pytest.raises(ValueError):
            apply_contrast_stretching(image=sample_bgr_image, alpha=alpha)

    @pytest.mark.parametrize("beta", [-1, 256, 1000, 5.5, "10"])
    def test_out_of_range_beta_raises(self, sample_bgr_image, beta):
        with pytest.raises(ValueError):
            apply_contrast_stretching(image=sample_bgr_image, beta=beta)
