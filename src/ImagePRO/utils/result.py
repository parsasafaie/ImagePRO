
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, List

import csv
import cv2
import numpy as np


ImageArrayLike = Union[np.ndarray, List[np.ndarray]]


@dataclass
class Result:
    """
    Unified container for outputs of ImagePRO operations.

    Stores optional image(s), structured data, and arbitrary metadata.
    Provides helpers to save image(s) or CSV outputs.
    Designed for consistent return types across all ImagePRO functions.

    Attributes:
        image (Optional[ImageArrayLike]):
            A single numpy array or list of arrays representing image(s).
            For BGR images by default, matching OpenCV convention.
        data (Optional[Any]):
            Structured data like landmark points [x, y, z], detection results
            [class, confidence, bbox], or other operation-specific data.
        meta (dict[str, Any]):
            Dictionary of metadata (e.g., processing parameters, confidence scores).

    Example:
        >>> result = detect_faces(image)  # Returns Result
        >>> result.save_as_img('output.jpg')  # Save annotated image
        >>> result.save_as_csv('faces.csv')   # Save detection data
        >>> print(f"Found {len(result.data)} faces")
        >>> print(f"Average confidence: {result.meta['mean_confidence']}")
    """

    image: Optional[ImageArrayLike] = field(default=None, repr=False)
    data: Optional[Any] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def save_as_img(self, path: str | Path) -> Result:
        """
        Save the contained image(s) to disk.

        Args:
            path (str | Path):
                Output file path. If a list of images, files are auto-suffixed.

        Returns:
            Result: Self, for method chaining.

        Raises:
            ValueError: If no image is present.
            TypeError: If path or image type is invalid.
            IOError: If saving fails.
        """
        if self.image is None:
            raise ValueError("No image to save in this result.")

        # Validate path type
        if not isinstance(path, (str, Path)):
            raise TypeError("'path' must be a string or pathlib.Path.")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate image(s) type
        if not isinstance(self.image, (np.ndarray, list)):
            raise TypeError("'image' must be a NumPy array or list of arrays.")
        if isinstance(self.image, list) and not all(isinstance(i, np.ndarray) for i in self.image):
            raise TypeError("All items in the image list must be NumPy arrays.")

        # Save single image
        if isinstance(self.image, np.ndarray):
            ok = cv2.imwrite(str(out_path), self.image)
            if not ok:
                raise IOError(f"Failed to save image: {out_path}")
            return self

        # Save list of images with auto-suffixes
        base = str(out_path)
        for idx, img in enumerate(self.image):
            if idx == 0:
                path_i = base
            else:
                # Suffix logic: .jpg → _{idx}.jpg, else _{idx} before extension
                if base.endswith(".jpg"):
                    path_i = base.replace(".jpg", f"_{idx}.jpg")
                else:
                    p = Path(base)
                    if p.suffix:
                        path_i = str(p.with_name(f"{p.stem}_{idx}{p.suffix}"))
                    else:
                        path_i = f"{base}_{idx}"
            ok = cv2.imwrite(path_i, img)
            if not ok:
                raise IOError(f"Failed to save image: {path_i}")
        return self

    def save_as_csv(
        self,
        path: str | Path,
        *,
        rows: Optional[list[list[Any]]] = None
    ) -> Result:
        """
        Save structured data as a CSV file.

        Args:
            path (str | Path):
                Output CSV file path.
            rows (Optional[list[list[Any]]], optional):
                Data to write as rows. If None, uses self.data. Defaults to None.

        Returns:
            Result: Self, for method chaining.

        Raises:
            ValueError: If no data is available to save.
            TypeError: If path type is invalid.
            IOError: If writing fails.
        """
        payload = rows if rows is not None else self.data
        if payload is None:
            raise ValueError("No data available to save as CSV.")
        if not isinstance(path, (str, Path)):
            raise TypeError("'path' must be a string or pathlib.Path.")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if isinstance(payload, (list, tuple)):
                    writer.writerows(flatten_rows(payload))
                else:
                    writer.writerow([payload])
        except Exception as exc:
            raise IOError(f"Failed to write CSV to {out_path}: {exc}") from exc
        return self

def flatten_rows(data):
    """Flatten any nested lists/tuples into rows suitable for CSV writing."""
    rows = []
    for item in data:
        if isinstance(item, (list, tuple)):
            if item and all(isinstance(x, (list, tuple)) for x in item):
                rows.extend(flatten_rows(item))
            else:
                rows.append(list(item))
        else:
            rows.append([item])
    return rows
