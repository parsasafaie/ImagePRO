"""Unit tests for ImagePRO.pre_processing.sharpen."""

from __future__ import annotations

import numpy as np
import pytest

from ImagePRO.pre_processing.sharpen import (
    apply_laplacian_sharpening,
    apply_unsharp_masking,
)


class TestLaplacianSharpening:
    def test_constant_image_unchanged(self):
        from ImagePRO.utils.image import Image

        image = Image.from_array(np.full((12, 12, 3), 90, np.uint8))
        result = apply_laplacian_sharpening(image=image, coefficient=5.0)
        assert np.array_equal(result.image, image._data)

    def test_zero_coefficient_returns_input(self, sample_bgr_array, sample_bgr_image):
        result = apply_laplacian_sharpening(image=sample_bgr_image, coefficient=0)
        assert np.array_equal(result.image, sample_bgr_array)

    def test_shape_dtype_preserved(self, sample_bgr_image):
        result = apply_laplacian_sharpening(image=sample_bgr_image, coefficient=3.0)
        assert result.image.shape == sample_bgr_image.shape
        assert result.image.dtype == np.uint8

    def test_meta_contents(self, sample_bgr_image):
        result = apply_laplacian_sharpening(image=sample_bgr_image, coefficient=2.5)
        assert result.data is None
        assert result.meta["operation"] == "apply_laplacian_sharpening"
        assert result.meta["coefficient"] == 2.5

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            apply_laplacian_sharpening(image=np.zeros((4, 4, 3)))

    @pytest.mark.parametrize("coefficient", [-0.1, -5, "3"])
    def test_invalid_coefficient_raises(self, sample_bgr_image, coefficient):
        with pytest.raises(ValueError):
            apply_laplacian_sharpening(image=sample_bgr_image, coefficient=coefficient)


class TestUnsharpMasking:
    def test_constant_image_unchanged(self):
        from ImagePRO.utils.image import Image

        image = Image.from_array(np.full((12, 12, 3), 90, np.uint8))
        result = apply_unsharp_masking(image=image, coefficient=1.0)
        assert np.array_equal(result.image, image._data)

    def test_zero_coefficient_returns_input(self, sample_bgr_array, sample_bgr_image):
        result = apply_unsharp_masking(image=sample_bgr_image, coefficient=0)
        assert np.array_equal(result.image, sample_bgr_array)

    def test_shape_dtype_preserved(self, sample_bgr_image):
        result = apply_unsharp_masking(image=sample_bgr_image, coefficient=1.0)
        assert result.image.shape == sample_bgr_image.shape
        assert result.image.dtype == np.uint8

    def test_meta_contents(self, sample_bgr_image):
        result = apply_unsharp_masking(image=sample_bgr_image, coefficient=0.8)
        assert result.meta["operation"] == "apply_unsharp_masking"
        assert result.meta["coefficient"] == 0.8

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            apply_unsharp_masking(image=[1, 2, 3])

    @pytest.mark.parametrize("coefficient", [-1.0, 0 - 1, "1"])
    def test_invalid_coefficient_raises(self, sample_bgr_image, coefficient):
        with pytest.raises(ValueError):
            apply_unsharp_masking(image=sample_bgr_image, coefficient=coefficient)
