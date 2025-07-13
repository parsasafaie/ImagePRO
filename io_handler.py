# io_handler.py

import cv2
import numpy as np
import os
import json
import csv
from pathlib import Path


class IOHandler:
    @staticmethod
    def load_image(image_path=None, np_image=None):
        """
        Loads an image from file or uses the provided NumPy array.

        Parameters:
            image_path (str | None): Path to input image file.
            np_image (np.ndarray | None): Pre-loaded image array.

        Returns:
            np.ndarray: Loaded image as a NumPy array.

        Raises:
            TypeError: If inputs are of incorrect type.
            FileNotFoundError: If image file not found.
            ValueError: If both sources are None or image loading fails.
        """
        if image_path is not None:
            if not isinstance(image_path, str):
                raise TypeError("'image_path' must be a string.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found at '{image_path}'")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image at '{image_path}'")
            return img
        elif np_image is not None:
            if not isinstance(np_image, np.ndarray):
                raise TypeError("'np_image' must be a NumPy array.")
            return np_image
        else:
            raise ValueError("At least one of 'image_path' or 'np_image' must be provided.")

    @staticmethod
    def save_image(np_image, result_path=None):
        """
        Saves an image to file or returns it directly.

        Parameters:
            np_image (np.ndarray): Image to save or return.
            result_path (str | None): Path to save the image. If None, returns the image array.

        Returns:
            str | np.ndarray: Confirmation message if saved, or the image array if not.

        Raises:
            TypeError: If inputs are of incorrect type.
            IOError: If saving the image fails.
        """
        if not isinstance(np_image, np.ndarray):
            raise TypeError("'np_image' must be a NumPy array.")

        if result_path is not None:
            if not isinstance(result_path, str):
                raise TypeError("'result_path' must be a string or None.")

            success = cv2.imwrite(result_path, np_image)
            if not success:
                raise IOError(f"Failed to save image at '{result_path}'")
            return f"Image saved at {result_path}"
        else:
            return np_image

    @staticmethod
    def save_csv(data, result_path=None):
        """
        Saves data as CSV file or returns it directly.

        Parameters:
            data (list[list]): Data to save or return.
            result_path (str | None): Path to save the CSV. If None, returns the data.

        Returns:
            str | list[list]: Confirmation message if saved, or the data if not.

        Raises:
            TypeError: If inputs are of incorrect type.
        """
        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            raise TypeError("'data' must be a list of lists.")

        if result_path is not None:
            if not isinstance(result_path, str):
                raise TypeError("'result_path' must be a string or None.")

            with open(result_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)
            return f"CSV saved at {result_path}"
        else:
            return data

    @staticmethod
    def save_json(data, result_path=None):
        """
        Saves data as JSON file or returns it directly.

        Parameters:
            data (any): JSON-serializable data to save or return.
            result_path (str | None): Path to save the JSON. If None, returns the data.

        Returns:
            str | any: Confirmation message if saved, or the data if not.

        Raises:
            TypeError: If inputs are of incorrect type.
        """
        if result_path is not None:
            if not isinstance(result_path, str):
                raise TypeError("'result_path' must be a string or None.")

            with open(result_path, "w") as f:
                json.dump(data, f, indent=4)
            return f"JSON saved at {result_path}"
        else:
            return data

    @staticmethod
    def save(data, result_path=None, file_type=None):
        """
        Automatically selects the correct save method based on file extension or `file_type`.

        Parameters:
            data (any): Data to save or return.
            result_path (str | None): Path to save the data. If None, returns the data.
            file_type (str | None): Force file type ('image', 'csv', 'json'). If None, detects by extension.

        Returns:
            str | any: Confirmation message if saved, or the data if not.

        Raises:
            ValueError: If file type or extension is unsupported.
            TypeError: If inputs are of incorrect type.
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
                raise ValueError(f"Unsupported file extension '{ext}' for 'result_path'")
        
        if file_type == "image":
            return IOHandler.save_image(data, result_path)
        elif file_type == "csv":
            return IOHandler.save_csv(data, result_path)
        elif file_type == "json":
            return IOHandler.save_json(data, result_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")