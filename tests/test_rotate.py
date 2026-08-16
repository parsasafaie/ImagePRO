"""Unit tests for ImagePRO.pre_processing.rotate."""

from __future__ import annotations

import numpy as np
import pytest

from ImagePRO.pre_processing.rotate import (
    rotate_image_180,
    rotate_image_270,
    rotate_image_90,
    rotate_image_custom,
)


class TestFixedRotations:
    def test_rotate_90_clockwise(self, sample_bgr_array, sample_bgr_image):
        result = rotate_image_90(image=sample_bgr_image)
        assert np.array_equal(result.image, np.rot90(sample_bgr_array, k=-1))

    def test_rotate_180(self, sample_bgr_array, sample_bgr_image):
        result = rotate_image_180(image=sample_bgr_image)
        assert np.array_equal(result.image, np.rot90(sample_bgr_array, k=2))

    def test_rotate_270_clockwise_equals_90_counterclockwise(
        self, sample_bgr_array, sample_bgr_image
    ):
        result = rotate_image_270(image=sample_bgr_image)
        assert np.array_equal(result.image, np.rot90(sample_bgr_array, k=1))

    @pytest.mark.parametrize(
        "func,angle",
        [(rotate_image_90, 90), (rotate_image_180, 180), (rotate_image_270, 270)],
    )
    def test_meta_contents(self, sample_bgr_image, func, angle):
        result = func(image=sample_bgr_image)
        assert result.data is None
        assert result.meta["operation"] == func.__name__
        assert result.meta["angle"] == angle
        assert result.meta["source"] is sample_bgr_image

    @pytest.mark.parametrize(
        "func",
        [rotate_image_90, rotate_image_180, rotate_image_270],
    )
    def test_non_image_raises(self, func):
        with pytest.raises(TypeError):
            func(image=np.zeros((4, 4, 3)))


class TestCustomRotation:
    def test_zero_angle_is_identity(self, sample_bgr_array, sample_bgr_image):
        result = rotate_image_custom(image=sample_bgr_image, angle=0.0)
        assert np.array_equal(result.image, sample_bgr_array)

    def test_full_turn_is_identity(self, sample_bgr_image):
        result = rotate_image_custom(image=sample_bgr_image, angle=360.0)
        assert result.image.shape == sample_bgr_image.shape

    def test_shape_preserved(self, sample_bgr_image):
        result = rotate_image_custom(image=sample_bgr_image, angle=30.0, scale=1.5)
        assert result.image.shape == sample_bgr_image.shape

    def test_meta_contents(self, sample_bgr_image):
        result = rotate_image_custom(image=sample_bgr_image, angle=45.0, scale=2.0)
        assert result.meta["operation"] == "rotate_image_custom"
        assert result.meta["angle"] == 45.0
        assert result.meta["scale"] == 2.0

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            rotate_image_custom(image="img", angle=30.0)

    def test_non_numeric_angle_raises(self, sample_bgr_image):
        with pytest.raises(TypeError):
            rotate_image_custom(image=sample_bgr_image, angle="45")

    @pytest.mark.parametrize("scale", [0, -1.0])
    def test_non_positive_scale_raises(self, sample_bgr_image, scale):
        with pytest.raises(ValueError):
            rotate_image_custom(image=sample_bgr_image, angle=30.0, scale=scale)

    def test_non_numeric_scale_raises(self, sample_bgr_image):
        with pytest.raises(ValueError):
            rotate_image_custom(image=sample_bgr_image, angle=30.0, scale="big")
