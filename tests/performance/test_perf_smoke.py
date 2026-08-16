"""Performance smoke tests.

These are coarse timing guards that catch gross regressions (e.g., an
accidental per-pixel Python loop). They are NOT benchmarks: bounds are
deliberately generous and machine-dependent. Run with:

    pytest --run-performance -m perf

No results should be quoted as measurements; they only assert that
operations finish within a sane budget.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ImagePRO.pre_processing.blur import apply_average_blur, apply_gaussian_blur
from ImagePRO.pre_processing.grayscale import convert_to_grayscale
from ImagePRO.pre_processing.resize import resize_image
from ImagePRO.pre_processing.rotate import rotate_image_90
from ImagePRO.utils.image import Image

pytestmark = pytest.mark.perf

# Generous wall-clock budgets for a 2048x2048 image on modest hardware.
BUDGET_SECONDS = 10.0


@pytest.fixture(scope="module")
def large_image():
    rng = np.random.default_rng(42)
    array = rng.integers(0, 256, size=(2048, 2048, 3), dtype=np.uint8)
    return Image.from_array(array)


def _timed(operation):
    start = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - start
    return result, elapsed


def test_grayscale_large_image(large_image):
    _, elapsed = _timed(lambda: convert_to_grayscale(image=large_image))
    assert elapsed < BUDGET_SECONDS


def test_average_blur_large_image(large_image):
    _, elapsed = _timed(lambda: apply_average_blur(image=large_image))
    assert elapsed < BUDGET_SECONDS


def test_gaussian_blur_large_image(large_image):
    _, elapsed = _timed(lambda: apply_gaussian_blur(image=large_image))
    assert elapsed < BUDGET_SECONDS


def test_rotate_large_image(large_image):
    _, elapsed = _timed(lambda: rotate_image_90(image=large_image))
    assert elapsed < BUDGET_SECONDS


def test_resize_large_image(large_image):
    _, elapsed = _timed(
        lambda: resize_image(image=large_image, new_size=(1024, 1024))
    )
    assert elapsed < BUDGET_SECONDS


def test_repeated_operations_stay_fast(large_image):
    """A loop of small ops should not degrade between iterations."""
    budgets = []
    for _ in range(3):
        _, elapsed = _timed(lambda: rotate_image_90(image=large_image))
        budgets.append(elapsed)
    assert max(budgets) < BUDGET_SECONDS
