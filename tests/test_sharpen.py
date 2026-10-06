"""Unit tests for ImagePRO.pre_processing.sharpen."""

from __future__ import annotations

import numpy as np
import pytest

from ImagePRO.pre_processing.sharpen import (
    apply_laplacian_sharpening,
    apply_unsharp_masking,
)
from ImagePRO.utils.image import Image


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

    def test_step_edge_is_preserved_not_saturated(self):
        # Regression: casting |laplacian| to uint8 before combining used to
        # truncate edge responses > 255. A one-pixel bright stripe on a dark
        # background gives |laplacian| = 400 at the stripe (4*230 - 4*30),
        # which the old code truncated to 144 and turned the stripe into
        # 174 instead of clipping it at 255.
        image = Image.from_array(
            np.concatenate(
                [np.full((8, 8, 3), 30, np.uint8),
                 np.full((8, 1, 3), 230, np.uint8),
                 np.full((8, 7, 3), 30, np.uint8)],
                axis=1,
            )
        )
        result = apply_laplacian_sharpening(image=image, coefficient=1.0)
        # Flat regions must survive sharpening unchanged
        assert np.array_equal(result.image[:, :7], np.full((8, 7, 3), 30, np.uint8))
        assert np.array_equal(result.image[:, 10:], np.full((8, 6, 3), 30, np.uint8))
        # The stripe must clip at white (truncated laplacian gave 174)
        assert (result.image[:, 8] == 255).all()
        # The flanks get the un-truncated overshoot (200 + 30)
        assert (result.image[:, 7] == 230).all()
        assert (result.image[:, 9] == 230).all()
        assert result.image.dtype == np.uint8

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

    def test_negative_mask_side_contributes(self):
        # Regression: the old cv2.subtract + addWeighted chain saturated
        # negative (original < blurred) mask values to 0 and then BRIGHTENED
        # flat regions (img + c*blurred) instead of sharpening. A one-pixel
        # bright stripe has a negative mask on its dark flanks; those must
        # undershoot below the base level rather than stay at it.
        image = Image.from_array(
            np.concatenate(
                [np.full((8, 8, 3), 30, np.uint8),
                 np.full((8, 1, 3), 230, np.uint8),
                 np.full((8, 7, 3), 30, np.uint8)],
                axis=1,
            )
        )
        result = apply_unsharp_masking(image=image, coefficient=1.0)
        # Flat regions far from the stripe must survive unchanged (not
        # brightened); the box blur reaches 2px, so undershoot spreads to
        # columns 6..10.
        assert np.array_equal(result.image[:, :6], np.full((8, 6, 3), 30, np.uint8))
        assert np.array_equal(result.image[:, 11:], np.full((8, 5, 3), 30, np.uint8))
        # The stripe overshoots and clips at white
        assert (result.image[:, 8] == 255).all()
        # The flanks undershoot (negative mask) and clip at black
        assert (result.image[:, 6:8] == 0).all()
        assert (result.image[:, 9:11] == 0).all()
        assert result.image.dtype == np.uint8
