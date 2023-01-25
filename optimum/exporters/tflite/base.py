# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorFlow Lite configuration base classes."""

from abc import ABC, abstractmethod
from ctypes import ArgumentError
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from transformers.utils import is_tf_available


if is_tf_available():
    import tensorflow as tf

from ..base import ExportConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, TFPreTrainedModel

    from ...utils import DummyInputGenerator

    if is_tf_available():
        from tensorflow import TensorSpec


class MissingMandatoryAxisDimension(ValueError):
    pass


class TFLiteConfig(ExportConfig, ABC):
    """
    Base class for TFLite exportable model describing metadata on how to export the model through the TFLite format.

    Class attributes:

    - NORMALIZED_CONFIG_CLASS (`Type`) -- A class derived from [`~optimum.utils.NormalizedConfig`] specifying how to
    normalize the model config.
    - DUMMY_INPUT_GENERATOR_CLASSES (`Tuple[Type]`) -- A tuple of classes derived from
    [`~optimum.utils.DummyInputGenerator`] specifying how to create dummy inputs.
    - ATOL_FOR_VALIDATION (`Union[float, Dict[str, float]]`) -- A float or a dictionary mapping task names to float,
    where the float values represent the absolute tolerance value to use during model conversion validation.

    Args:
        config (`transformers.PretrainedConfig`):
            The model configuration.
        task (`str`, defaults to `"default"`):
            The task the model should be exported for.

        The rest of the arguments are used to specify the shape of the inputs the model can take.
        They are required or not depending on the model the `TFLiteConfig` is designed for.
    """

    NORMALIZED_CONFIG_CLASS = None
    DUMMY_INPUT_GENERATOR_CLASSES = ()
    ATOL_FOR_VALIDATION: Union[float, Dict[str, float]] = 1e-5
    MANDATORY_AXES = ()

    _TASK_TO_COMMON_OUTPUTS = {
        "causal-lm": ["logits"],
        "default": ["last_hidden_state"],
        "image-classification": ["logits"],
        "image-segmentation": ["logits", "pred_boxes", "pred_masks"],
        "masked-im": ["logits"],
        "masked-lm": ["logits"],
        "multiple-choice": ["logits"],
        "object-detection": ["logits", "pred_boxes"],
        "question-answering": ["start_logits", "end_logits"],
        "semantic-segmentation": ["logits"],
        "seq2seq-lm": ["logits", "encoder_last_hidden_state"],
        "sequence-classification": ["logits"],
        "token-classification": ["logits"],
        "speech2seq-lm": ["logits"],
        "audio-classification": ["logits"],
        "audio-frame-classification": ["logits"],
        "audio-ctc": ["logits"],
        "audio-xvector": ["logits"],
    }

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str,
        batch_size: int = 1,
        sequence_length: Optional[int] = None,
        num_choices: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_channels: Optional[int] = None,
        feature_size: Optional[int] = None,
        nb_max_frames: Optional[int] = None,
    ):
        self._config = config
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
        self.mandatory_axes = ()
        self.task = task
        self._axes: Dict[str, int] = {}

        # To avoid using **kwargs.
        axes_values = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "num_choices": num_choices,
            "width": width,
            "height": height,
            "num_channels": num_channels,
            "feature_size": feature_size,
            "nb_max_frames": nb_max_frames,
        }
        for name, value in axes_values.items():
            setattr(self, name, value)

    def _update_mandatory_axes(self):
        axes = []
        for axis in self.MANDATORY_AXES:
            if isinstance(axis, tuple):
                tasks, name = axis
                if not isinstance(tasks, tuple):
                    tasks = (tasks,)
                if self.task not in tasks:
                    continue
            else:
                name = axis
            axes.append(name)
        self.mandatory_axes = tuple(axes)

    @property
    def task(self) -> str:
        return self._task

    @task.setter
    def task(self, value: str):
        self._task = value
        self._update_mandatory_axes()

    def __getattr__(self, attr_name) -> Any:
        if attr_name in self._axes:
            return self._axes[attr_name]
        else:
            raise AttributeError(attr_name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.MANDATORY_AXES:
            if value is None:
                if self._normalized_config.has_attribute(name):
                    value = getattr(self._normalized_config, name)
            self._axes[name] = value
        else:
            return super().__setattr__(name, value)

    def _validate_mandatory_axes(self):
        for name, axis_dim in self._axes.items():
            if axis_dim is None:
                raise MissingMandatoryAxisDimension(
                    f"The value for the {name} axis is missing, it is needed to perform the export to TensorFlow Lite."
                )

    def _create_dummy_input_generator_classes(self) -> List["DummyInputGenerator"]:
        self._validate_mandatory_axes()
        return [cls_(self.task, self._normalized_config, **self._axes) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES]

    @property
    def values_override(self) -> Optional[Dict[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting.

        Returns:
            `Optional[Dict[str, Any]]`: A dictionary specifying the configuration items to override.
        """
        if hasattr(self._config, "use_cache"):
            return {"use_cache": False}

        return None

    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        """
        List containing the names of the inputs the exported model should take.

        Returns:
            `List[str]`: A list of input names.
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> List[str]:
        """
        List containing the names of the outputs the exported model should have.

        Returns:
            `List[str]`: A list of output names.
        """
        return self._TASK_TO_COMMON_OUTPUTS[self.task]

    def generate_dummy_inputs(self) -> Dict[str, "tf.Tensor"]:
        """
        Generates dummy inputs that the exported model should be able to process.
        This method is actually used to determine the input specs that are needed for the export.

        Returns:
            `Dict[str, tf.Tensor]`: A dictionary mapping input names to dummy tensors.
        """
        dummy_inputs_generators = self._create_dummy_input_generator_classes()
        dummy_inputs = {}

        for input_name in self.inputs:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework="tf")
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy inputs for "{input_name}". Try adding a proper dummy input generator '
                    "to the model TFLite config."
                )

        return dummy_inputs

    @property
    def inputs_specs(self) -> List["TensorSpec"]:
        """
        List containing the input specs for each input name in self.inputs.

        Returns:
            `List[tf.TensorSpec]`: A list of tensor specs.
        """
        dummy_inputs = self.generate_dummy_inputs()
        return [
            tf.TensorSpec(dummy_input.shape, dtype=dummy_input.dtype, name=input_name)
            for input_name, dummy_input in dummy_inputs.items()
        ]

    def model_to_signatures(
        self, model: "TFPreTrainedModel", **model_kwargs: Any
    ) -> Dict[str, "tf.types.experimental.ConcreteFunction"]:
        """
        Creates the signatures that will be used when exporting the model to a `tf.SavedModel`.
        Each signature can be used to perform inference on the model for a given set of inputs.

        Auto-encoder models have only one signature, decoder models can have two, one for the decoder without
        caching, and one for the decoder with caching, seq2seq models can have three, and so on.
        """
        input_names = self.inputs
        output_names = self.outputs

        def forward(*args):
            if len(args) != len(input_names):
                raise ArgumentError(
                    f"The number of inputs provided ({len(args)} do not match the number of expected inputs: "
                    f"{', '.join(input_names)}."
                )
            kwargs = dict(zip(input_names, args))
            outputs = model.call(**kwargs, **model_kwargs)
            return {key: value for key, value in outputs.items() if key in output_names}

        function = tf.function(forward, input_signature=self.inputs_specs).get_concrete_function()

        return {"model": function}
