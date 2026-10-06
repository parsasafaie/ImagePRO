"""Unit tests for the dataset generator (camera-free paths only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ImagePRO.pre_processing.dataset_generator import capture_bulk_pictures


class TestValidation:
    @pytest.mark.parametrize("num_images", [0, -1, 1.5, "10"])
    def test_invalid_num_images_raises(self, tmp_path, num_images):
        with pytest.raises(ValueError):
            capture_bulk_pictures(tmp_path / "out", num_images=num_images)

    @pytest.mark.parametrize("start_index", [-1, 1.5, None])
    def test_invalid_start_index_raises(self, tmp_path, start_index):
        with pytest.raises(ValueError):
            capture_bulk_pictures(tmp_path / "out", start_index=start_index)

    @pytest.mark.parametrize("min_confidence", [-0.1, 1.1, "0.7"])
    def test_invalid_confidence_raises(self, tmp_path, min_confidence):
        with pytest.raises(ValueError):
            capture_bulk_pictures(tmp_path / "out", min_confidence=min_confidence)

    @pytest.mark.parametrize("delay", [-0.5, "1"])
    def test_invalid_delay_raises(self, tmp_path, delay):
        with pytest.raises(ValueError):
            capture_bulk_pictures(tmp_path / "out", delay=delay)


class TestFolderHandling:
    def test_existing_folder_raises_file_exists_error(self, tmp_path):
        face_dir = tmp_path / "dataset" / "alice"
        face_dir.mkdir(parents=True)
        with pytest.raises(FileExistsError):
            capture_bulk_pictures(tmp_path / "dataset", "alice", num_images=1)

    def test_face_id_subfolder_created(self, tmp_path, monkeypatch):
        class ClosedCamera:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return False

        monkeypatch.setattr(
            "ImagePRO.pre_processing.dataset_generator.cv2.VideoCapture",
            ClosedCamera,
        )
        with pytest.raises(RuntimeError):
            capture_bulk_pictures(tmp_path / "dataset", "bob", num_images=1)
        assert (tmp_path / "dataset" / "bob").exists()


class TestCameraFailure:
    def test_inaccessible_camera_raises_runtime_error(self, tmp_path, monkeypatch):
        class ClosedCamera:
            def __init__(self, *args, **kwargs):
                self.index = args[0] if args else kwargs.get("index")

            def isOpened(self):
                return False

        monkeypatch.setattr(
            "ImagePRO.pre_processing.dataset_generator.cv2.VideoCapture",
            ClosedCamera,
        )
        with pytest.raises(RuntimeError, match="Cannot access camera"):
            capture_bulk_pictures(tmp_path / "dataset", "carol", num_images=1)


class TestValidationOrder:
    def test_invalid_apply_resize_raises_before_camera_or_folder(
        self, tmp_path, monkeypatch
    ):
        # Regression: apply_resize was only validated inside the capture
        # loop, after the folder was created and the camera opened.
        created = []

        class OpenCamera:
            def __init__(self, *args, **kwargs):
                created.append("camera")

            def isOpened(self):
                return True

        monkeypatch.setattr(
            "ImagePRO.pre_processing.dataset_generator.cv2.VideoCapture",
            OpenCamera,
        )
        with pytest.raises(ValueError):
            capture_bulk_pictures(
                tmp_path / "dataset", "dave", num_images=1, apply_resize=(0, 0)
            )
        with pytest.raises(ValueError):
            capture_bulk_pictures(
                tmp_path / "dataset", "dave", num_images=1, apply_resize=True
            )
        assert created == []  # camera never touched
        assert not (tmp_path / "dataset" / "dave").exists()  # folder untouched

    def test_bool_parameters_raise(self, tmp_path):
        with pytest.raises(ValueError):
            capture_bulk_pictures(tmp_path / "out", num_images=True)
        with pytest.raises(ValueError):
            capture_bulk_pictures(tmp_path / "out", min_confidence=True)
        with pytest.raises(ValueError):
            capture_bulk_pictures(tmp_path / "out", delay=True)


class TestNoFaceGiveUp:
    def test_gives_up_after_max_empty_frames(self, tmp_path, monkeypatch, capsys):
        # Regression: frames without a face used to be skipped forever, so
        # an empty scene blocked the generator indefinitely.
        import numpy as np

        import mediapipe as mp  # real package or the conftest stub

        from ImagePRO.pre_processing import dataset_generator as dg
        from ImagePRO.utils.result import Result

        class FakeCamera:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((10, 10, 3), np.uint8)

            def release(self):
                pass

        monkeypatch.setattr(
            "ImagePRO.pre_processing.dataset_generator.cv2.VideoCapture",
            FakeCamera,
        )
        monkeypatch.setattr(
            mp.solutions.face_mesh, "FaceMesh", lambda **kwargs: object()
        )
        monkeypatch.setattr(
            "ImagePRO.pre_processing.dataset_generator.detect_faces",
            lambda **kwargs: Result(
                image=None,
                data=None,
                meta={"error": "No face landmarks detected"},
            ),
        )
        monkeypatch.setattr(dg.time, "sleep", lambda *_: None)
        monkeypatch.setattr(dg, "MAX_EMPTY_FRAMES", 5)

        # No exception: the loop breaks and reports the give-up.
        capture_bulk_pictures(tmp_path / "dataset", "erin", num_images=5, delay=0)
        captured = capsys.readouterr()
        assert "No face detected" in captured.out
        assert not any((tmp_path / "dataset" / "erin").glob("*.jpg"))

    def test_camera_failure_also_gives_up(self, tmp_path, monkeypatch):
        import mediapipe as mp  # real package or the conftest stub

        from ImagePRO.pre_processing import dataset_generator as dg

        class DyingCamera:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def read(self):
                return False, None

            def release(self):
                pass

        monkeypatch.setattr(
            "ImagePRO.pre_processing.dataset_generator.cv2.VideoCapture",
            DyingCamera,
        )
        monkeypatch.setattr(
            mp.solutions.face_mesh, "FaceMesh", lambda **kwargs: object()
        )
        monkeypatch.setattr(dg, "MAX_EMPTY_FRAMES", 3)

        with pytest.raises(RuntimeError, match="stopped delivering frames"):
            capture_bulk_pictures(tmp_path / "dataset", "frank", num_images=5, delay=0)
