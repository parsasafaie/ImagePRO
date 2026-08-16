"""Unit tests for ImagePRO.pre_processing.resize."""

from __future__ import annotations

import numpy as np
import pytest

from ImagePRO.pre_processing.resize import resize_image


class TestResizeImage:
    def test_resizes_to_requested_dimensions(self, sample_bgr_image):
        result = resize_image(image=sample_bgr_image, new_size=(8, 4))
        # new_size is (width, height); array shape is (height, width).
        assert result.image.shape == (4, 8, 3)

    def test_upscale(self, sample_bgr_image):
        result = resize_image(image=sample_bgr_image, new_size=(64, 48))
        assert result.image.shape == (48, 64, 3)

    def test_constant_image_stays_constant(self):
        from ImagePRO.utils.image import Image

        image = Image.from_array(np.full((10, 10, 3), 77, np.uint8))
        result = resize_image(image=image, new_size=(25, 15))
        assert np.all(result.image == 77)

    def test_meta_contents(self, sample_bgr_image):
        result = resize_image(image=sample_bgr_image, new_size=(16, 12))
        assert result.data is None
        assert result.meta["operation"] == "resize_image"
        assert result.meta["new_size"] == (16, 12)
        assert result.meta["source"] is sample_bgr_image

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            resize_image(image=None, new_size=(8, 8))

    @pytest.mark.parametrize(
        "size",
        [(0, 10), (10, 0), (-5, 10), (10,), (10, 10, 10), [10, 10], (10.5, 10), ("10", 10)],
    )
    def test_invalid_size_raises(self, sample_bgr_image, size):
        with pytest.raises(ValueError):
            resize_image(image=sample_bgr_image, new_size=size)
