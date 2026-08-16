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
