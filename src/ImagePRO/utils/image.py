
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Literal

import cv2
import numpy as np


Colorspace = Literal["BGR", "RGB", "GRAY"]
SourceType = Literal["path", "array"]


@dataclass
class Image:
    """
    Lightweight, immutable image wrapper for ImagePRO.

    Use factory constructors to create instances. All methods return new Image objects.
    Images are handled in BGR format by default to match OpenCV conventions.

    Attributes:
        _data (np.ndarray):
            The underlying image data (NumPy array).
        path (Optional[Path]):
            Source file path if loaded from disk, else None.
        colorspace (Colorspace):
            Image colorspace ("BGR", "RGB", or "GRAY"). Defaults to "BGR".
        source_type (SourceType):
            Indicates if image was loaded from "path" or "array".

    Example:
        >>> # Load from file (default BGR)
        >>> img = Image.from_path('input.jpg', colorspace="BGR")
        >>> print(img.shape)  # (H, W, 3)
        >>> 
        >>> # Or wrap a NumPy array
        >>> img = Image.from_array(np_array, colorspace="RGB")
    """

    _data: np.ndarray = field(repr=False)
    path: Optional[Path] = None
    colorspace: Colorspace = "BGR"
    source_type: SourceType = "array"

    @classmethod
    def from_path(
        cls, 
        path: str | Path,
        colorspace: Colorspace = "BGR"
    ) -> Image:
        """
        Create an Image instance from a file path.

        Args:
            path (str | Path):
                Path to the image file.
            colorspace (Colorspace, optional):
                Colorspace of the image array in that path ("BGR", "RGB", or "GRAY"). Defaults to "BGR".

        Returns:
            Image: New Image instance loaded from disk.

        Raises:
            TypeError: If path is not str or Path.
            ValueError: If image cannot be loaded.
            ValueError: If colorspace is invalid.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError("'path' must be a string or pathlib.Path.")
        np_image = cv2.imread(str(path))
        if np_image is None:
            raise ValueError(f"Failed to load image from {path}")
        if colorspace not in ("BGR", "RGB", "GRAY"):
            raise ValueError("'colorspace' must be one of 'BGR', 'RGB', 'GRAY'.")
        return cls(_data=np_image, path=Path(path), colorspace=colorspace, source_type="path")

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        colorspace: Colorspace = "BGR"
    ) -> Image:
        """
        Create an Image instance from a NumPy array.

        Args:
            array (np.ndarray):
                Image data as a NumPy array.
            colorspace (Colorspace, optional):
                Colorspace of the array ("BGR", "RGB", or "GRAY"). Defaults to "BGR".

        Returns:
            Image: New Image instance wrapping the array.

        Raises:
            TypeError: If array is not a NumPy array.
            ValueError: If colorspace is invalid.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("'array' must be a numpy.ndarray.")
        if colorspace not in ("BGR", "RGB", "GRAY"):
            raise ValueError("'colorspace' must be one of 'BGR', 'RGB', 'GRAY'.")
        return cls(_data=array, path=None, colorspace=colorspace, source_type="array")

    @property
    def shape(self) -> Tuple[int, int] | Tuple[int, int, int]:
        """
        Returns the shape of the image array.

        Returns:
            Tuple[int, int] | Tuple[int, int, int]: Image shape.
        """
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        """
        Returns the dtype of the image array.

        Returns:
            np.dtype: Data type of the image array.
        """
        return self._data.dtype
        