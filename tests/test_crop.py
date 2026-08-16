"""Unit tests for ImagePRO.pre_processing.crop."""

from __future__ import annotations

import numpy as np
import pytest

from ImagePRO.pre_processing.crop import crop_image


class TestCropImage:
    def test_cropped_region_matches_numpy_slice(self, sample_bgr_array, sample_bgr_image):
        result = crop_image(
            image=sample_bgr_image, start_point=(3, 2), end_point=(9, 6)
        )
        assert np.array_equal(result.image, sample_bgr_array[2:6, 3:9])

    def test_full_image_crop(self, sample_bgr_array, sample_bgr_image):
        height, width = sample_bgr_image.shape[:2]
        result = crop_image(
            image=sample_bgr_image, start_point=(0, 0), end_point=(width, height)
        )
        assert np.array_equal(result.image, sample_bgr_array)

    def test_meta_contents(self, sample_bgr_image):
        result = crop_image(
            image=sample_bgr_image, start_point=(1, 2), end_point=(8, 10)
        )
        assert result.data is None
        assert result.meta["operation"] == "crop_image"
        assert result.meta["start_point"] == (1, 2)
        assert result.meta["end_point"] == (8, 10)
        assert result.meta["source"] is sample_bgr_image

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            crop_image(image=np.zeros((5, 5, 3)), start_point=(0, 0), end_point=(2, 2))

    @pytest.mark.parametrize(
        "start,end",
        [
            ((0, 0), (0, 5)),   # zero width
            ((0, 0), (5, 0)),   # zero height
            ((5, 5), (2, 9)),   # x2 <= x1
            ((5, 5), (9, 2)),   # y2 <= y1
            ((-1, 0), (5, 5)),  # negative x1
            ((0, -1), (5, 5)),  # negative y1
        ],
    )
    def test_invalid_geometry_raises(self, sample_bgr_image, start, end):
        with pytest.raises(ValueError):
            crop_image(image=sample_bgr_image, start_point=start, end_point=end)

    def test_out_of_bounds_raises(self, sample_bgr_image):
        height, width = sample_bgr_image.shape[:2]
        with pytest.raises(ValueError):
            crop_image(
                image=sample_bgr_image,
                start_point=(0, 0),
                end_point=(width + 1, height),
            )
        with pytest.raises(ValueError):
            crop_image(
                image=sample_bgr_image,
                start_point=(0, 0),
                end_point=(width, height + 1),
            )

    @pytest.mark.parametrize(
        "start,end",
        [
            ([0, 0], [5, 5]),          # lists instead of tuples
            ((0, 0, 0), (5, 5)),       # wrong length
            ((0, 0), (5, 5, 5)),       # wrong length
            ((0, 0.5), (5, 5)),        # non-int coordinate
            ((0, 0), ("5", 5)),        # non-int coordinate
            (None, (5, 5)),            # not a tuple
        ],
    )
    def test_non_tuple_or_non_int_coordinates_raise(self, sample_bgr_image, start, end):
        with pytest.raises(TypeError):
            crop_image(image=sample_bgr_image, start_point=start, end_point=end)
