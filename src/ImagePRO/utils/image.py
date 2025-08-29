from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import cv2

from .io_handler import IOHandler
from ..pre_processing.resize import resize_image


Colorspace = Literal["BGR", "RGB", "GRAY"]
SourceType = Literal["path", "array"]


@dataclass
class Image:
    """Lightweight image wrapper with fluent, immutable transforms.

    Use factory constructors (from_path, from_array) to create instances.
    Methods return new Image instances; the original remains unchanged.
    """

    _data: np.ndarray = field(repr=False)
    path: Optional[Path] = None
    colorspace: Colorspace = "BGR"
    source_type: SourceType = "array"

    # ---- Factory constructors -------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path) -> "Image":
        if not isinstance(path, (str, Path)):
            raise TypeError("'path' must be a string or pathlib.Path")
        np_image = cv2.imread(path)
        if np_image is None:
            raise ValueError(f"Failed to load image from {path}")
        return cls(_data=np_image, path=Path(path), colorspace="BGR", source_type="path")

    @classmethod
    def from_array(cls, array: np.ndarray, colorspace: Colorspace = "BGR") -> "Image":
        if not isinstance(array, np.ndarray):
            raise TypeError("'array' must be a numpy.ndarray")
        if colorspace not in ("BGR", "RGB", "GRAY"):
            raise ValueError("'colorspace' must be one of 'BGR', 'RGB', 'GRAY'")
        return cls(_data=array, path=None, colorspace=colorspace, source_type="array")

    # ---- Introspection --------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int] | Tuple[int, int, int]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype
        