"""Unit tests for ImagePRO.pre_processing.histogram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ImagePRO.pre_processing.histogram import show_histogram
from ImagePRO.utils.image import Image


class TestShowHistogram:
    def test_bgr_image_returns_result_with_plot(self, sample_bgr_image):
        result = show_histogram(sample_bgr_image)
        assert result.image is None
        assert result.data is plt
        assert result.meta["operation"] == "show_histogram"
        assert result.meta["source"] is sample_bgr_image

    def test_rgb_image_returns_result(self, sample_rgb_image):
        result = show_histogram(sample_rgb_image)
        assert result.data is plt

    def test_gray_image_returns_result(self, sample_gray_image):
        result = show_histogram(sample_gray_image)
        assert result.data is plt

    @pytest.mark.parametrize("colorspace", ["BGR", "RGB", "GRAY"])
    def test_creates_figure(self, sample_bgr_array, colorspace):
        plt.close("all")
        image = Image.from_array(sample_bgr_array, colorspace=colorspace)
        show_histogram(image)
        assert plt.gcf() is not None
        assert len(plt.get_fignums()) == 1

    def test_non_image_raises(self):
        with pytest.raises(TypeError):
            show_histogram(np.zeros((4, 4, 3)))

    def test_unknown_colorspace_raises(self, sample_bgr_array):
        # Bypass factory validation to reach the unknown-colorspace branch.
        image = Image(_data=sample_bgr_array, colorspace="YUV")
        with pytest.raises(ValueError):
            show_histogram(image)


class TestNoFigureLeak:
    def test_figure_is_closed_after_return(self, sample_bgr_image):
        # Regression: every call created a figure that was never closed, so
        # batch use accumulated figures in memory.
        plt.close("all")
        show_histogram(sample_bgr_image)
        assert len(plt.get_fignums()) == 0

    def test_gray_image_gets_labels(self, sample_gray_image, monkeypatch):
        # Regression: grayscale histograms had no title/labels at all.
        # show_histogram closes its figure before returning, so the labels
        # are captured the moment the function closes the figure.
        import matplotlib.pyplot as pyplot

        captured = {}
        orig_close = pyplot.close

        def spy_close(fig=None):
            if fig is None and pyplot.get_fignums():
                target = pyplot.gcf()
                if target.axes and "title" not in captured:
                    captured["title"] = target.axes[0].get_title()
                    captured["xlabel"] = target.axes[0].get_xlabel()
            return orig_close(fig)

        monkeypatch.setattr(pyplot, "close", spy_close)
        pyplot.close("all")
        show_histogram(sample_gray_image)
        assert len(pyplot.get_fignums()) == 0  # closed, not leaked
        assert captured["title"] == "Histogram of GRAY Channels"
        assert captured["xlabel"] == "Pixel Intensity"
        pyplot.close("all")
