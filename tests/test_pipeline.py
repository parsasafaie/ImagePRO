"""Unit tests for ImagePRO.pipeline."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from ImagePRO.pipeline import Pipeline, Step
from ImagePRO.pre_processing.blur import apply_gaussian_blur
from ImagePRO.pre_processing.grayscale import convert_to_grayscale
from ImagePRO.pre_processing.resize import resize_image
from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result


@pytest.fixture
def multiply_function():
    """Custom operation following the Image -> Result convention."""

    def function(image, *, scale=1):
        output = image._data if scale == 1 else np.clip(image._data * scale, 0, 255).astype(np.uint8)
        return Result(image=output, meta={"operation": "multiply", "scale": scale})

    return function


@pytest.fixture
def colorspace_recorder():
    """Custom operation recording the colorspace of every Image received."""
    seen = []

    def function(image):
        seen.append(image.colorspace)
        return Result(image=image._data)

    function.seen = seen
    return function


class TestStep:
    def test_forwards_kwargs_to_function(self, sample_bgr_image):
        step = Step(apply_gaussian_blur, kernel_size=(9, 9))
        result = step(sample_bgr_image)
        assert result.meta["kernel_size"] == (9, 9)

    def test_works_without_kwargs(self, sample_bgr_image):
        step = Step(apply_gaussian_blur)
        result = step(sample_bgr_image)
        assert result.meta["kernel_size"] == (5, 5)

    def test_name_property(self):
        assert Step(apply_gaussian_blur).name == "apply_gaussian_blur"

    def test_name_falls_back_to_type_name(self):
        step = Step(partial(apply_gaussian_blur, kernel_size=(3, 3)))
        assert step.name == "partial"

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            Step("not callable")

    def test_repr_shows_name_and_kwargs(self):
        step = Step(apply_gaussian_blur, kernel_size=(7, 7))
        assert "apply_gaussian_blur" in repr(step)
        assert "kernel_size=(7, 7)" in repr(step)


class TestConstruction:
    def test_from_list(self):
        pipeline = Pipeline([apply_gaussian_blur, convert_to_grayscale])
        assert len(pipeline) == 2

    def test_from_tuple(self):
        pipeline = Pipeline((apply_gaussian_blur, convert_to_grayscale))
        assert len(pipeline) == 2

    def test_from_varargs(self):
        pipeline = Pipeline(apply_gaussian_blur, convert_to_grayscale)
        assert len(pipeline) == 2

    def test_from_single_callable(self):
        pipeline = Pipeline(apply_gaussian_blur)
        assert len(pipeline) == 1

    def test_from_single_step(self):
        pipeline = Pipeline(Step(apply_gaussian_blur, kernel_size=(7, 7)))
        assert len(pipeline) == 1
        assert pipeline.operations[0].kwargs == {"kernel_size": (7, 7)}

    def test_mixed_callables_and_steps(self):
        pipeline = Pipeline(
            Step(apply_gaussian_blur, kernel_size=(3, 3)),
            convert_to_grayscale,
        )
        assert len(pipeline) == 2
        assert isinstance(pipeline.operations[0], Step)
        assert isinstance(pipeline.operations[1], Step)

    def test_empty_pipeline_is_allowed(self):
        assert len(Pipeline()) == 0

    def test_plain_callables_are_wrapped_in_steps(self):
        pipeline = Pipeline(apply_gaussian_blur)
        assert isinstance(pipeline.operations[0], Step)
        assert pipeline.operations[0].function is apply_gaussian_blur

    def test_operations_property_returns_tuple(self):
        pipeline = Pipeline(apply_gaussian_blur, convert_to_grayscale)
        assert isinstance(pipeline.operations, tuple)
        assert [step.name for step in pipeline.operations] == [
            "apply_gaussian_blur",
            "convert_to_grayscale",
        ]

    @pytest.mark.parametrize("bad", [42, "blur", None, 3.14])
    def test_non_callable_operation_raises(self, bad):
        with pytest.raises(TypeError):
            Pipeline(bad)

    def test_non_callable_inside_list_raises(self):
        with pytest.raises(TypeError):
            Pipeline([apply_gaussian_blur, 42])


class TestApply:
    def test_returns_new_pipeline(self):
        original = Pipeline(apply_gaussian_blur)
        extended = original.apply(convert_to_grayscale)
        assert extended is not original
        assert len(original) == 1
        assert len(extended) == 2

    def test_is_chainable(self):
        pipeline = (
            Pipeline()
            .apply(apply_gaussian_blur, kernel_size=(3, 3))
            .apply(convert_to_grayscale)
            .apply(resize_image, new_size=(16, 12))
        )
        assert len(pipeline) == 3
        assert [step.name for step in pipeline] == [
            "apply_gaussian_blur",
            "convert_to_grayscale",
            "resize_image",
        ]

    def test_forwards_kwargs(self, sample_bgr_image):
        pipeline = Pipeline().apply(apply_gaussian_blur, kernel_size=(9, 9))
        result = pipeline.run(sample_bgr_image)
        assert result.meta["step_meta"][0]["kernel_size"] == (9, 9)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            Pipeline().apply(42)


class TestRun:
    def test_single_step_returns_result(self, sample_bgr_image):
        result = Pipeline(apply_gaussian_blur).run(sample_bgr_image)
        assert isinstance(result, Result)
        assert result.image.shape == sample_bgr_image.shape
        assert result.image.dtype == np.uint8

    def test_multi_step_matches_manual_application(self, sample_bgr_image):
        pipeline = Pipeline([
            apply_gaussian_blur,
            convert_to_grayscale,
            Step(resize_image, new_size=(16, 12)),
        ])
        result = pipeline.run(sample_bgr_image)

        blurred = apply_gaussian_blur(image=sample_bgr_image).image
        gray = convert_to_grayscale(
            image=Image.from_array(blurred, colorspace="BGR")
        ).image
        resized = resize_image(
            image=Image.from_array(gray, colorspace="GRAY"), new_size=(16, 12)
        ).image
        assert np.array_equal(result.image, resized)

    def test_meta_contents(self, sample_bgr_image):
        pipeline = Pipeline([
            Step(apply_gaussian_blur, kernel_size=(7, 7)),
            convert_to_grayscale,
        ])
        result = pipeline.run(sample_bgr_image)
        assert result.meta["operation"] == "pipeline"
        assert result.meta["steps"] == ["apply_gaussian_blur", "convert_to_grayscale"]
        assert len(result.meta["step_meta"]) == 2
        assert result.meta["step_meta"][0]["kernel_size"] == (7, 7)
        assert result.meta["step_meta"][1]["operation"] == "convert_to_grayscale"
        assert result.meta["source"] is sample_bgr_image

    def test_final_step_meta_is_merged(self, sample_bgr_image):
        result = Pipeline(
            Step(apply_gaussian_blur, kernel_size=(7, 7))
        ).run(sample_bgr_image)
        # Pipeline keys win over the final step's own metadata
        assert result.meta["operation"] == "pipeline"
        assert result.meta["source"] is sample_bgr_image
        # The final step's own keys survive the merge
        assert result.meta["kernel_size"] == (7, 7)

    def test_input_as_ndarray(self, sample_bgr_array):
        result = Pipeline(apply_gaussian_blur).run(sample_bgr_array)
        assert result.image.shape == sample_bgr_array.shape

    def test_input_as_2d_ndarray_is_treated_as_gray(self, sample_gray_image):
        result = Pipeline(apply_gaussian_blur).run(sample_gray_image._data)
        assert result.meta["source"].colorspace == "GRAY"

    def test_input_as_path(self, image_file):
        result = Pipeline(convert_to_grayscale).run(image_file)
        assert result.image.ndim == 2

    def test_input_as_str_path(self, image_file):
        result = Pipeline(convert_to_grayscale).run(str(image_file))
        assert result.image.ndim == 2

    def test_invalid_input_type_raises(self, sample_bgr_image):
        pipeline = Pipeline(apply_gaussian_blur)
        with pytest.raises(TypeError):
            pipeline.run(123)
        with pytest.raises(TypeError):
            pipeline.run(None)
        with pytest.raises(TypeError):
            pipeline.run([sample_bgr_image])

    def test_empty_pipeline_raises(self, sample_bgr_image):
        with pytest.raises(ValueError):
            Pipeline().run(sample_bgr_image)

    def test_pipeline_is_reusable_across_inputs(self, multiply_function):
        pipeline = Pipeline(multiply_function)
        first = pipeline.run(Image.from_array(np.full((8, 8, 3), 10, np.uint8)))
        second = pipeline.run(Image.from_array(np.full((12, 6, 3), 200, np.uint8)))
        assert len(pipeline) == 1
        assert first.image.shape == (8, 8, 3)
        assert second.image.shape == (12, 6, 3)

    def test_reruns_do_not_interfere(self, sample_bgr_image):
        pipeline = Pipeline(apply_gaussian_blur)
        first = pipeline.run(sample_bgr_image)
        second = pipeline.run(sample_bgr_image)
        assert np.array_equal(first.image, second.image)
        assert first.image is not second.image

    def test_custom_operation_returning_ndarray(self, sample_bgr_image):
        def double(image):
            return np.clip(image._data * 2, 0, 255).astype(np.uint8)

        result = Pipeline(double).run(sample_bgr_image)
        assert isinstance(result, Result)
        assert result.meta["steps"] == ["double"]
        expected = np.clip(sample_bgr_image._data * 2, 0, 255).astype(np.uint8)
        assert np.array_equal(result.image, expected)

    def test_unsupported_operation_return_raises(self, sample_bgr_image):
        def bad_operation(image):
            return "not an image or result"

        with pytest.raises(TypeError):
            Pipeline(bad_operation).run(sample_bgr_image)

    def test_non_final_step_without_image_raises(self, sample_bgr_image):
        def data_only(image):
            return Result(image=None, data=[1, 2, 3])

        pipeline = Pipeline(data_only, apply_gaussian_blur)
        with pytest.raises(ValueError):
            pipeline.run(sample_bgr_image)

    def test_final_step_may_omit_image(self, sample_bgr_image):
        def data_only(image):
            return Result(image=None, data=[1, 2, 3])

        result = Pipeline(data_only).run(sample_bgr_image)
        assert result.image is None
        assert result.data == [1, 2, 3]
        assert result.meta["steps"] == ["data_only"]

    def test_final_step_may_return_image_list(self, sample_bgr_image):
        def two_images(image):
            return Result(image=[image._data, image._data.copy()])

        result = Pipeline(two_images).run(sample_bgr_image)
        assert isinstance(result.image, list)
        assert len(result.image) == 2

    def test_non_final_step_with_image_list_raises(self, sample_bgr_image):
        def two_images(image):
            return Result(image=[image._data, image._data.copy()])

        pipeline = Pipeline(two_images, apply_gaussian_blur)
        with pytest.raises(ValueError):
            pipeline.run(sample_bgr_image)

    def test_colorspace_propagates_through_chain(
        self, sample_bgr_image, colorspace_recorder
    ):
        Pipeline(convert_to_grayscale, colorspace_recorder).run(sample_bgr_image)
        assert colorspace_recorder.seen == ["GRAY"]

    def test_bgr_colorspace_propagates(self, sample_bgr_image, colorspace_recorder):
        Pipeline(apply_gaussian_blur, colorspace_recorder).run(sample_bgr_image)
        assert colorspace_recorder.seen == ["BGR"]

    def test_operation_validation_errors_propagate(self, sample_gray_image):
        # convert_to_grayscale refuses already-gray images; the pipeline
        # must surface that error unchanged.
        with pytest.raises(ValueError):
            Pipeline(convert_to_grayscale).run(sample_gray_image)

    def test_nested_pipeline_runs_as_single_step(self, sample_bgr_image):
        inner = Pipeline(apply_gaussian_blur)
        outer = Pipeline(inner, convert_to_grayscale)
        result = outer.run(sample_bgr_image)
        assert result.image.ndim == 2
        assert result.meta["steps"][0] == "Pipeline"

    def test_source_meta_is_original_input(self, image_file):
        image = Image.from_path(image_file)
        result = Pipeline(apply_gaussian_blur).run(image)
        assert result.meta["source"] is image


class TestCall:
    def test_call_is_shorthand_for_run(self, sample_bgr_image):
        pipeline = Pipeline(apply_gaussian_blur)
        by_call = pipeline(sample_bgr_image)
        by_run = pipeline.run(sample_bgr_image)
        assert np.array_equal(by_call.image, by_run.image)
        assert by_call.meta["operation"] == "pipeline"


class TestDunders:
    def test_len(self):
        assert len(Pipeline()) == 0
        assert len(Pipeline(apply_gaussian_blur, convert_to_grayscale)) == 2

    def test_iter_yields_steps(self):
        pipeline = Pipeline(apply_gaussian_blur, convert_to_grayscale)
        steps = list(pipeline)
        assert all(isinstance(step, Step) for step in steps)
        assert [step.name for step in steps] == [
            "apply_gaussian_blur",
            "convert_to_grayscale",
        ]

    def test_repr_lists_step_names(self):
        pipeline = Pipeline(apply_gaussian_blur, convert_to_grayscale)
        assert repr(pipeline) == "Pipeline(apply_gaussian_blur -> convert_to_grayscale)"

    def test_repr_of_empty_pipeline(self):
        assert repr(Pipeline()) == "Pipeline(<empty>)"
