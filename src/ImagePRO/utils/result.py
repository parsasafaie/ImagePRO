from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union, List

import numpy as np
import cv2
import csv
from pathlib import Path


ImageArrayLike = Union[np.ndarray, List[np.ndarray]]


@dataclass
class Result:
    """Unified result object for ImagePRO operations.

    - Holds an optional image (numpy array) OR list of arrays, and optional structured data
    - Provides helpers to save image(s) or CSV outputs
    - Designed to be returned by functional APIs (e.g., resize(Image, ...))
    """

    image: Optional[ImageArrayLike] = field(default=None, repr=False)
    data: Optional[Any] = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- Introspection --------------------------------------------------------

    def to_numpy(self, *, copy: bool = False) -> np.ndarray:
        if self.image is None:
            raise ValueError("This result does not contain an image.")
        if isinstance(self.image, list):
            return [image.copy() if copy else image for image in self.image] 
        return self.image.copy() if copy else self.image

    # ---- Saving helpers (IOHandler-compatible validation) ---------------------
    def save_as_img(self, path: str | Path) -> "Result":
        if self.image is None:
            raise ValueError("No image to save in this result.")

        # Validate path
        if not isinstance(path, (str, Path)):
            raise TypeError("'path' must be a string or pathlib.Path.")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate image(s)
        if not isinstance(self.image, (np.ndarray, list)):
            raise TypeError("'image' must be a NumPy array or list of arrays.")
        if isinstance(self.image, list) and not all(isinstance(i, np.ndarray) for i in self.image):
            raise TypeError("All items in the image list must be NumPy arrays.")

        if isinstance(self.image, np.ndarray):
            ok = cv2.imwrite(str(out_path), self.image)
            if not ok:
                raise IOError(f"Failed to save image: {out_path}")
            return self

        # List case: save with suffixes based on base path, mimicking IOHandler behavior
        base = str(out_path)
        for idx, img in enumerate(self.image):
            if idx == 0:
                path_i = base
            else:
                # naive replacement matching IOHandler behavior (replace .jpg only)
                if base.endswith(".jpg"):
                    path_i = base.replace(".jpg", f"_{idx}.jpg")
                else:
                    # If not .jpg, append index before extension if present, else at end
                    p = Path(base)
                    if p.suffix:
                        path_i = str(p.with_name(f"{p.stem}_{idx}{p.suffix}"))
                    else:
                        path_i = f"{base}_{idx}"
            ok = cv2.imwrite(path_i, img)
            if not ok:
                raise IOError(f"Failed to save image: {path_i}")
        return self

    def save_as_csv(self, path: str | Path, *, rows: Optional[list[list[Any]]] = None) -> "Result":
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
                if isinstance(payload, list) and payload and all(isinstance(i, (list, tuple)) for i in payload):
                    writer.writerows(payload)  # list of rows
                elif isinstance(payload, (list, tuple)):
                    writer.writerow(list(payload))  # single row
                else:
                    writer.writerow([payload]) # scalar
        except Exception as exc:
            raise IOError(f"Failed to write CSV to {out_path}: {exc}") from exc
        return self
