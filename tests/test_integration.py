"""Integration tests exercising complete ImagePRO workflows.

These pipelines only use base dependencies (OpenCV/NumPy) so they run
everywhere; analysis-module pipelines are covered by their mocked unit
tests.
"""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from ImagePRO.pre_processing.blur import apply_gaussian_blur
from ImagePRO.pre_processing.contrast import apply_contrast_stretching
from ImagePRO.pre_processing.crop import crop_image
from ImagePRO.pre_processing.grayscale import convert_to_grayscale
from ImagePRO.pre_processing.resize import resize_image
from ImagePRO.pre_processing.rotate import rotate_image_90
from ImagePRO.utils.image import Image


class TestFullPipeline:
    def test_load_enhance_resize_save_roundtrip(self, image_file, tmp_path):
        image = Image.from_path(image_file)
        gray = convert_to_grayscale(image=image)
        enhanced = apply_contrast_stretching(image=image, alpha=1.2, beta=8)
        blurred = apply_gaussian_blur(image=image, kernel_size=(5, 5))
        resized = resize_image(image=Image.from_array(blurred.image), new_size=(16, 12))

        assert gray.image.ndim == 2
        assert enhanced.image.ndim == 2
        assert resized.image.shape == (12, 16, 3)

        out = tmp_path / "pipeline.jpg"
        resized.save_as_img(out)
        reloaded = cv2.imread(str(out))
        assert reloaded.shape == (12, 16, 3)

    def test_crop_then_rotate_then_save(self, sample_bgr_array, tmp_path):
        image = Image.from_array(sample_bgr_array.copy())
        cropped = crop_image(image=image, start_point=(2, 2), end_point=(20, 16))
        rotated = rotate_image_90(image=Image.from_array(cropped.image))

        expected = np.rot90(sample_bgr_array[2:16, 2:20], k=-1)
        assert np.array_equal(rotated.image, expected)

        out = tmp_path / "crop_rotate.png"
        rotated.save_as_img(out)
        assert np.array_equal(cv2.imread(str(out)), expected)

    def test_pipeline_meta_tracks_sources(self, sample_bgr_image):
        blurred = apply_gaussian_blur(image=sample_bgr_image)
        cropped = crop_image(
            image=Image.from_array(blurred.image), start_point=(0, 0), end_point=(10, 10)
        )
        assert cropped.meta["source"]._data.shape == sample_bgr_image.shape
        assert blurred.meta["source"] is sample_bgr_image


class TestCsvWorkflow:
    def test_landmark_style_data_saved_and_reloaded(self, tmp_path):
        data = [
            [0, 10, 0.25, 0.5, -0.1],
            [0, 11, 0.30, 0.6, -0.2],
        ]
        out = tmp_path / "landmarks.csv"
        from ImagePRO.utils.result import Result

        Result(data=data).save_as_csv(out)
        with out.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows == [
            ["0", "10", "0.25", "0.5", "-0.1"],
            ["0", "11", "0.3", "0.6", "-0.2"],
        ]

    def test_image_and_csv_output_in_one_result(self, sample_bgr_image, tmp_path):
        result = apply_gaussian_blur(image=sample_bgr_image)
        img_path = tmp_path / "img" / "out.png"
        csv_path = tmp_path / "data" / "out.csv"
        result.save_as_img(img_path).save_as_csv(csv_path, rows=[[1, 2, 3]])
        assert img_path.exists() and csv_path.exists()


class TestImageSourceCompatibility:
    def test_same_operation_from_path_and_array(self, image_file, sample_bgr_array):
        from_path = Image.from_path(image_file)
        from_array = Image.from_array(sample_bgr_array.copy())
        result_path = convert_to_grayscale(image=from_path)
        result_array = convert_to_grayscale(image=from_array)
        assert np.array_equal(result_path.image, result_array.image)

    def test_error_paths_across_pipeline(self, sample_bgr_image):
        with pytest.raises(ValueError):
            resize_image(image=sample_bgr_image, new_size=(0, 10))
        with pytest.raises(TypeError):
            crop_image(image=sample_bgr_image, start_point=[0, 0], end_point=[2, 2])
        with pytest.raises(ValueError):
            apply_gaussian_blur(image=sample_bgr_image, kernel_size=(4, 4))
