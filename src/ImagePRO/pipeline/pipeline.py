from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterator

# Add src directory to path for absolute imports
_file_path = Path(__file__).resolve()
_src_path = _file_path.parents[2]  # Go up to src directory
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import numpy as np

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


class Step:
    """
    A single operation in a Pipeline, with optional keyword arguments.

    Wraps any callable that accepts an Image as its first positional argument
    and returns a Result (or a numpy.ndarray, treated as the output image).
    Keyword arguments given to the constructor are forwarded to the wrapped
    function on every call, so a Step fully describes one stage of a pipeline.

    Example:
        >>> from ImagePRO.pre_processing.blur import apply_gaussian_blur
        >>> step = Step(apply_gaussian_blur, kernel_size=(7, 7))
        >>> result = step(image)  # Same as apply_gaussian_blur(image, kernel_size=(7, 7))

    Attributes:
        function (Callable):
            The wrapped function to execute.
        kwargs (dict[str, Any]):
            Keyword-only arguments forwarded to the function on each call.
    """

    def __init__(self, function: Callable[..., Result], **kwargs: Any) -> None:
        """
        Create a Step from a function and its keyword arguments.

        Args:
            function (Callable):
                Function to wrap. Must be callable and accept an Image as its
                first positional argument.
            **kwargs (Any):
                Keyword-only arguments forwarded to the function on every call.

        Raises:
            TypeError: If function is not callable.
        """
        if not callable(function):
            raise TypeError("'function' must be a callable.")
        self.function = function
        self.kwargs: dict[str, Any] = dict(kwargs)

    @property
    def name(self) -> str:
        """
        Human-readable name of the wrapped function, used in pipeline metadata.

        Returns:
            str: The function's __name__ when available, otherwise the name
            of its type (e.g. 'partial' for functools.partial objects).
        """
        return getattr(self.function, "__name__", type(self.function).__name__)

    def __call__(self, image: Image) -> Result:
        """
        Execute the wrapped function on an image.

        Args:
            image (Image):
                Input image passed to the wrapped function.

        Returns:
            Result: Whatever the wrapped function returns.

        Raises:
            TypeError: If image is not an Image instance (raised by the
                wrapped function's own validation).
            ValueError: If wrapped function arguments are invalid (raised by
                the wrapped function's own validation).
        """
        return self.function(image, **self.kwargs)

    def __repr__(self) -> str:
        """
        Return a readable representation of the step.

        Returns:
            str: Representation including the function name and arguments.
        """
        args = ", ".join(f"{key}={value!r}" for key, value in self.kwargs.items())
        return f"Step({self.name}{', ' + args if args else ''})"


class Pipeline:
    """
    A reusable, ordered chain of image processing operations.

    A Pipeline is defined once from a sequence of functions and can then be
    run on any input as many times as needed. Each operation receives an
    Image and its output image is automatically wrapped and fed to the next
    operation; the final Result is returned with pipeline metadata attached.

    Operations are the standard ImagePRO functions (which take an Image and
    return a Result) or any custom callable following the same convention.
    A custom function may also return a plain numpy.ndarray, which is treated
    as the output image. Per-operation keyword arguments are configured with
    Step, or fluently via apply().

    Pipelines are immutable: apply() returns a new Pipeline and leaves the
    original untouched, and run() never mutates the pipeline, so a single
    Pipeline instance is safe to reuse across many inputs.

    Example:
        >>> from ImagePRO.pre_processing.blur import apply_gaussian_blur
        >>> from ImagePRO.pre_processing.grayscale import convert_to_grayscale
        >>> from ImagePRO.pre_processing.resize import resize_image
        >>>
        >>> pipeline = Pipeline([
        ...     Step(apply_gaussian_blur, kernel_size=(7, 7)),
        ...     convert_to_grayscale,
        ...     Step(resize_image, new_size=(800, 600)),
        ... ])
        >>> result = pipeline.run("photo.jpg")   # Any Image, ndarray or path
        >>> result.save_as_img("processed.jpg")
    """

    def __init__(self, *operations: Any) -> None:
        """
        Create a Pipeline from one or more operations.

        Operations can be passed as separate arguments or as a single
        list/tuple. Passing another Pipeline as an operation nests it: the
        inner pipeline runs as a single step.

        Args:
            *operations (Callable | Step | list | tuple):
                Functions to execute in order. Each element must be callable;
                Step instances carry their keyword arguments. A single
                list/tuple of operations is also accepted.

        Raises:
            TypeError: If any operation is not callable.
        """
        if len(operations) == 1 and isinstance(operations[0], (list, tuple)):
            operations = tuple(operations[0])

        steps: list[Step] = []
        for operation in operations:
            if not callable(operation):
                raise TypeError(
                    "All pipeline operations must be callable; got "
                    f"{type(operation).__name__}. Wrap keyword arguments with "
                    "Step, e.g. Step(resize_image, new_size=(800, 600))."
                )
            steps.append(operation if isinstance(operation, Step) else Step(operation))
        self._operations = steps

    @property
    def operations(self) -> tuple[Step, ...]:
        """
        The ordered steps of this pipeline.

        Returns:
            tuple[Step, ...]: Step objects in execution order.
        """
        return tuple(self._operations)

    def apply(self, function: Callable[..., Result], **kwargs: Any) -> Pipeline:
        """
        Return a new Pipeline with one more operation appended.

        The current pipeline is not modified, following ImagePRO's
        non-destructive design principle.

        Args:
            function (Callable):
                Function to append. Must be callable and accept an Image as
                its first positional argument.
            **kwargs (Any):
                Keyword-only arguments forwarded to the function on every run.

        Returns:
            Pipeline: New pipeline containing all existing steps plus this one.

        Raises:
            TypeError: If function is not callable.

        Example:
            >>> pipeline = Pipeline(apply_gaussian_blur).apply(
            ...     resize_image, new_size=(800, 600)
            ... )
        """
        return Pipeline(*self._operations, Step(function, **kwargs))

    def run(self, image: Image | np.ndarray | str | Path) -> Result:
        """
        Execute all pipeline operations in order on the given input.

        The input is first normalized to an Image (file paths are loaded and
        NumPy arrays are wrapped, assuming BGR for 3-channel and GRAY for 2D
        arrays). Each operation then receives an Image built from the previous
        operation's output; 2D intermediates are tagged as GRAY and
        multi-channel ones keep the incoming colorspace.

        The returned Result carries the final operation's image and data. Its
        metadata contains the pipeline information (operation, steps,
        step_meta, source) merged over the final operation's own metadata.

        Args:
            image (Image | np.ndarray | str | Path):
                Input to process. Can be an Image instance, a NumPy array
                (BGR for 3-channel, GRAY for 2D), or a path to an image file.

        Returns:
            Result: Result object of the final operation.
                - image (np.ndarray | list[np.ndarray] | None): Final output image(s)
                - data (Any): Structured data of the final operation
                - meta (dict): Final operation metadata plus pipeline info:
                  'operation' ("pipeline"), 'steps' (list of step names),
                  'step_meta' (per-step metadata list), 'source' (input Image)

        Raises:
            TypeError: If image is not an Image, ndarray, or path.
            TypeError: If an operation returns an unsupported type.
            ValueError: If the pipeline has no operations.
            ValueError: If a non-final operation returns no image, so the
                chain cannot continue.
            TypeError/ValueError: As raised by the operations themselves for
                invalid inputs or arguments.
        """
        # Normalize input to an Image (cheap validation before any step runs)
        if isinstance(image, Image):
            current = image
        elif isinstance(image, np.ndarray):
            colorspace = "GRAY" if image.ndim == 2 else "BGR"
            current = Image.from_array(image, colorspace=colorspace)
        elif isinstance(image, (str, Path)):
            current = Image.from_path(image)
        else:
            raise TypeError(
                "'image' must be an Image instance, a numpy.ndarray, or a "
                "file path (str or pathlib.Path)."
            )

        if not self._operations:
            raise ValueError(
                "Cannot run an empty pipeline. Add operations at construction "
                "or with apply()."
            )

        source = current
        step_names: list[str] = []
        step_metas: list[dict[str, Any]] = []
        final_image: Any = None
        final_data: Any = None
        final_meta: dict[str, Any] = {}

        for index, step in enumerate(self._operations):
            is_last = index == len(self._operations) - 1
            output = step(current)

            # Normalize the step output (Result is the convention; a plain
            # ndarray from custom functions is accepted as the output image)
            if isinstance(output, Result):
                final_image = output.image
                final_data = output.data
                final_meta = dict(output.meta)
            elif isinstance(output, np.ndarray):
                final_image = output
                final_data = None
                final_meta = {}
            else:
                raise TypeError(
                    f"Step {index} ('{step.name}') returned "
                    f"{type(output).__name__}; pipeline operations must "
                    "return a Result object or a numpy.ndarray."
                )

            step_names.append(step.name)
            step_metas.append(final_meta)

            if is_last:
                break

            # Prepare the input of the next operation
            if final_image is None:
                raise ValueError(
                    f"Step {index} ('{step.name}') returned no image, so the "
                    "pipeline cannot continue. Only the final step of a "
                    "pipeline may return a result without an image."
                )
            if not isinstance(final_image, np.ndarray):
                raise ValueError(
                    f"Step {index} ('{step.name}') did not return a single "
                    "image array, so the pipeline cannot continue. Only the "
                    "final step of a pipeline may return multi-image results."
                )

            colorspace = "GRAY" if final_image.ndim == 2 else current.colorspace
            current = Image.from_array(final_image, colorspace=colorspace)

        meta = dict(final_meta)
        meta.update({
            "source": source,
            "operation": "pipeline",
            "steps": step_names,
            "step_meta": step_metas,
        })
        return Result(image=final_image, data=final_data, meta=meta)

    def __call__(self, image: Image | np.ndarray | str | Path) -> Result:
        """
        Run the pipeline on an input; shorthand for run().

        Args:
            image (Image | np.ndarray | str | Path):
                Input to process, as accepted by run().

        Returns:
            Result: Result object of the final operation.

        Raises:
            TypeError: If image is not an Image, ndarray, or path.
            ValueError: If the pipeline has no operations or a non-final
                operation returns no image.
        """
        return self.run(image)

    def __len__(self) -> int:
        """
        Return the number of operations in the pipeline.

        Returns:
            int: Number of steps.
        """
        return len(self._operations)

    def __iter__(self) -> Iterator[Step]:
        """
        Iterate over the pipeline's steps in execution order.

        Returns:
            Iterator[Step]: Iterator of Step objects.
        """
        return iter(self._operations)

    def __repr__(self) -> str:
        """
        Return a readable representation of the pipeline.

        Returns:
            str: Representation listing the step names in order.
        """
        if not self._operations:
            return "Pipeline(<empty>)"
        return f"Pipeline({' -> '.join(step.name for step in self._operations)})"
