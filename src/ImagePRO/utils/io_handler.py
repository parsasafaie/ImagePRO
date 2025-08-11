# io_handler.py

import os
import json
import csv
from pathlib import Path

import cv2
import numpy as np


class IOHandler:
    @staticmethod
    def load_image(image_path=None, np_image=None):
        """
        Load an image from disk or use an existing NumPy array.

        Args:
            image_path (str, optional): Path to the image file.
            np_image (np.ndarray, optional): Image as a NumPy array.

        Returns:
            np.ndarray: Loaded image.

        Raises:
            TypeError: Invalid input types.
            FileNotFoundError: File not found.
            ValueError: Both inputs are None or image loading failed.
        """
        if image_path is not None:
            if not isinstance(image_path, str):
                raise TypeError("'image_path' must be a string.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unable to load image: {image_path}")
            return img

        if np_image is not None:
            if not isinstance(np_image, np.ndarray):
                raise TypeError("'np_image' must be a NumPy array.")
            return np_image

        raise ValueError("Provide at least 'image_path' or 'np_image'.")

    @staticmethod
    def save_image(np_image, result_path=None):
        """
        Save a NumPy image array (or list of arrays) to disk or return it.

        Args:
            np_image (np.ndarray or list[np.ndarray]): Image(s) to save.
            result_path (str, optional): Save path.

        Returns:
            str or np.ndarray or list[np.ndarray]: 
                Save message if path given, otherwise returns input.

        Raises:
            TypeError: Invalid input types.
            IOError: Saving failed.
        """
        if not isinstance(np_image, (np.ndarray, list)):
            raise TypeError("'np_image' must be a NumPy array or list of arrays.")

        if isinstance(np_image, list) and not all(isinstance(i, np.ndarray) for i in np_image):
            raise TypeError("All items in the list must be NumPy arrays.")

        if result_path is not None and not isinstance(result_path, str):
            raise TypeError("'result_path' must be a string or None.")

        if result_path:
            if isinstance(np_image, np.ndarray):
                if not cv2.imwrite(result_path, np_image):
                    raise IOError(f"Failed to save image: {result_path}")
                return f"Image saved at {result_path}"

            for i, img in enumerate(np_image):
                suffix = f"_{i}" if i > 0 else ""
                path = result_path.replace(".jpg", f"{suffix}.jpg")
                if not cv2.imwrite(path, img):
                    raise IOError(f"Failed to save image: {path}")
            return f"Images saved using base path {result_path}"

        return "Skipped saving image because no file path was provided"

    @staticmethod
    def save_csv(data, result_path=None):
        """
        Save a list of lists as a CSV file or return it.

        Args:
            data (list[list]): Tabular data.
            result_path (str, optional): Save path.

        Returns:
            str or list[list]: Save message or original data.

        Raises:
            TypeError: Invalid input types.
        """
        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            raise TypeError("'data' must be a list of lists.")

        if result_path:
            if not isinstance(result_path, str):
                raise TypeError("'result_path' must be a string or None.")
            with open(result_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)
            return f"CSV saved at {result_path}"

        return "Skipped saving csv because no file path was provided"


    @staticmethod
    def save_json(data, result_path=None):
        """
        Save data as a JSON file or return it.

        Args:
            data (Any): JSON-serializable data.
            result_path (str, optional): Save path.

        Returns:
            str or Any: Save message or original data.

        Raises:
            TypeError: Invalid path type.
        """
        if result_path:
            if not isinstance(result_path, str):
                raise TypeError("'result_path' must be a string or None.")
            with open(result_path, "w") as f:
                json.dump(data, f, indent=4)
            return f"JSON saved at {result_path}"

        return "Skipped saving json because no file path was provided"


    @staticmethod
    def save(data, result_path=None, file_type=None):
        """
        Save data to disk automatically based on file type or extension.

        Args:
            data (Any): Data to save.
            result_path (str, optional): Save path.
            file_type (str, optional): 'image', 'csv', or 'json'.

        Returns:
            str or Any: Save message or original data.

        Raises:
            ValueError: Unsupported file type or extension.
            TypeError: Invalid input types.
        """
        if file_type is None:
            if result_path is None:
                return data

            ext = Path(result_path).suffix.lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                file_type = "image"
            elif ext == ".csv":
                file_type = "csv"
            elif ext == ".json":
                file_type = "json"
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        if file_type == "image":
            return IOHandler.save_image(data, result_path)
        elif file_type == "csv":
            return IOHandler.save_csv(data, result_path)
        elif file_type == "json":
            return IOHandler.save_json(data, result_path)

        raise ValueError(f"Unsupported file type: {file_type}")
