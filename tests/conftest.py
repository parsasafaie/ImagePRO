"""Shared pytest configuration and fixtures for the ImagePRO test suite.

Keeps the suite hermetic:

* Forces a non-interactive matplotlib backend so histogram tests never
  open windows.
* Makes the ``src`` layout importable without requiring an installed
  package.
* Installs lightweight stubs for the heavy optional dependencies
  (mediapipe, insightface, ultralytics) when they are not available,
  so tests run in base-only environments. The real packages are left
  untouched when present. Every test that needs a detector injects a
  fake via monkeypatching, so the stubs only have to satisfy the lazy
  imports inside ImagePRO's functions.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")

import numpy as np
import pytest


# Make the src layout importable without an installed package.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class _UnavailableModel:
    """Placeholder for optional-dependency models.

    Instantiating it fails loudly so tests remember to inject a fake
    detector instead of silently doing nothing.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Optional dependency is not installed; monkeypatch the "
            "detector class in the test."
        )


def _build_mediapipe_stub() -> types.ModuleType:
    mp = types.ModuleType("mediapipe")
    mp.__dict__.update(
        solutions=types.SimpleNamespace(
            face_mesh=types.SimpleNamespace(
                FaceMesh=_UnavailableModel,
                FACEMESH_TESSELATION=frozenset(),
            ),
            pose=types.SimpleNamespace(
                Pose=_UnavailableModel,
                POSE_CONNECTIONS=frozenset(),
            ),
            hands=types.SimpleNamespace(
                Hands=_UnavailableModel,
                HAND_CONNECTIONS=frozenset(),
            ),
            drawing_utils=types.SimpleNamespace(
                draw_landmarks=lambda *args, **kwargs: None,
            ),
            drawing_styles=types.SimpleNamespace(
                get_default_face_mesh_tesselation_style=lambda: None,
                get_default_pose_landmarks_style=lambda: None,
                get_default_hand_landmarks_style=lambda: None,
                get_default_hand_connections_style=lambda: None,
            ),
        )
    )
    return mp


def _build_ultralytics_stub() -> types.ModuleType:
    module = types.ModuleType("ultralytics")
    module.YOLO = _UnavailableModel
    return module


def _build_insightface_stubs() -> None:
    package = types.ModuleType("insightface")
    app = types.ModuleType("insightface.app")
    app.FaceAnalysis = _UnavailableModel
    package.app = app
    sys.modules["insightface"] = package
    sys.modules["insightface.app"] = app


def _install_optional_dependency_stubs() -> None:
    """Register import stubs for missing optional dependencies."""
    for module_name, builder in (
        ("mediapipe", _build_mediapipe_stub),
        ("ultralytics", _build_ultralytics_stub),
    ):
        if module_name not in sys.modules:
            try:
                __import__(module_name)
            except ImportError:
                sys.modules[module_name] = builder()

    try:
        __import__("insightface.app")
    except ImportError:
        _build_insightface_stubs()


_install_optional_dependency_stubs()

from ImagePRO.utils.image import Image  # noqa: E402


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after every test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def sample_bgr_array() -> np.ndarray:
    """Deterministic 32x24 BGR image with a smooth channel gradient."""
    height, width = 24, 32
    rng = np.arange(height * width, dtype=np.int32).reshape(height, width)
    blue = (rng * 7) % 256
    green = (rng * 13) % 256
    red = (rng * 29) % 256
    return np.stack([blue, green, red], axis=-1).astype(np.uint8)


@pytest.fixture
def sample_bgr_image(sample_bgr_array) -> Image:
    return Image.from_array(sample_bgr_array.copy(), colorspace="BGR")


@pytest.fixture
def sample_rgb_image(sample_bgr_array) -> Image:
    return Image.from_array(sample_bgr_array.copy(), colorspace="RGB")


@pytest.fixture
def sample_gray_image() -> Image:
    gray = (np.arange(12 * 10).reshape(12, 10) * 5 % 256).astype(np.uint8)
    return Image.from_array(gray, colorspace="GRAY")


@pytest.fixture
def image_file(tmp_path, sample_bgr_array) -> Path:
    """A real image file on disk, for from_path / save-reload tests."""
    path = tmp_path / "input.png"
    import cv2

    assert cv2.imwrite(str(path), sample_bgr_array)
    return path


def pytest_addoption(parser):
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance smoke tests (marked with 'perf').",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-performance"):
        return
    skip_perf = pytest.mark.skip(reason="needs --run-performance to run")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)
