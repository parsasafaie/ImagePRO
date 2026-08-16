"""Unit tests for ImagePRO.pre_processing.grayscale."""

from __future__ import annotations

import numpy as np
import pytest

from ImagePRO.pre_processing.grayscale import convert_to_grayscale


def solid_color_image(color):
    from ImagePRO.utils.image import Image

    return Image.from_array(
        np.tile(np.array(color, np.uint8), (10, 10, 1)), colorspace="BGR"
    )


class TestConvertToGrayscale:
    def test_white_becomes_white(self):
        image = solid_color_image((255, 255, 255))
        result = convert_to_grayscale(image=image)
        assert result.image.shape == (10, 10)
        assert np.all(result.image == 255)

    def test_black_becomes_black(self):
        image = solid_color_image((0, 0, 0))
        result = convert_to_grayscale(image=image)
        assert np.all(result.image == 0)

    def test_output_is_single_channel(self, sample_bgr_image):
        result = convert_to_grayscale(image=sample_bgr_image)
        assert result.image.ndim == 2
        assert result.image.shape == sample_bgr_image.shape[:2]
        assert result.image.dtype == np.uint8

    def test_bgr_and_rgb_labels_weight_channels_differently(self):
        # Pure red array: BGR weights red at 0.299, RGB weights it fully.
        array = np.zeros((6, 6, 3), np.uint8)
        array[..., 2] = 255  # last channel: R if BGR, B if RGB

        from ImagePRO.utils.image import Image

        bgr_result = convert_to_grayscale(image=Image.from_array(array, colorspace="BGR"))
        rgb_result = convert_to_grayscale(image=Image.from_array(array, colorspace="RGB"))
        assert not np.array_equal(bgr_result.image, rgb_result.image)

    def test_meta_contents(self, sample_bgr_image):
        result = convert_to_grayscale(image=sample_bgr_image)
        assert result.data is None
        assert result.meta["operation"] == "convert_to_grayscale"
        assert result.meta["source"] is sample_bgr_image

    def test_gray_input_raises(self, sample_gray_image):
        with pytest.raises(ValueError):
            convert_to_grayscale(image=sample_gray_image)

    @pytest.mark.parametrize("bad", [None, "img", np.zeros((4, 4, 3))])
    def test_non_image_raises(self, bad):
        with pytest.raises(TypeError):
            convert_to_grayscale(image=bad)
