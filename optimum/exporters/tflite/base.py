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
from collections import OrderedDict
from ctypes import ArgumentError
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# TODO: handle tensorflow dependency.
import tensorflow as tf
from transformers.utils import is_tf_available

from ..base import ExportConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, TFPreTrainedModel

    from ...utils import DummyInputGenerator

    if is_tf_available():
        from tensorflow import TensorSpec


class MissingMandatoryAxisDimension(ValueError):
    pass


class TFLiteConfig(ExportConfig, ABC):
    NORMALIZED_CONFIG_CLASS = None
    DUMMY_INPUT_GENERATOR_CLASSES = ()
    ATOL_FOR_VALIDATION: Union[float, Dict[str, float]] = 1e-5
    MANDATORY_AXES = ()

    _TASK_TO_COMMON_OUTPUTS = {
        "causal-lm": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "default": OrderedDict({"last_hidden_state": {0: "batch_size", 1: "sequence_length"}}),
        "image-classification": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "image-segmentation": OrderedDict(
            {
                "logits": {0: "batch_size", 1: "sequence_length"},
                "pred_boxes": {0: "batch_size", 1: "sequence_length"},
                "pred_masks": {0: "batch_size", 1: "sequence_length"},
            }
        ),
        "masked-im": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "masked-lm": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "multiple-choice": OrderedDict({"logits": {0: "batch_size"}}),
        "object-detection": OrderedDict(
            {
                "logits": {0: "batch_size", 1: "sequence_length"},
                "pred_boxes": {0: "batch_size", 1: "sequence_length"},
            }
        ),
        "question-answering": OrderedDict(
            {
                "start_logits": {0: "batch_size", 1: "sequence_length"},
                "end_logits": {0: "batch_size", 1: "sequence_length"},
            }
        ),
        "semantic-segmentation": OrderedDict({"logits": {0: "batch_size", 1: "num_labels", 2: "height", 3: "width"}}),
        "seq2seq-lm": OrderedDict(
            {
                "logits": {0: "batch_size", 1: "decoder_sequence_length"},
                "encoder_last_hidden_state": {0: "batch_size", 1: "encoder_sequence_length"},
            }
        ),
        "sequence-classification": OrderedDict({"logits": {0: "batch_size"}}),
        "token-classification": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "speech2seq-lm": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "audio-classification": OrderedDict({"logits": {0: "batch_size"}}),
        "audio-frame-classification": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "audio-ctc": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "audio-xvector": OrderedDict({"logits": {0: "batch_size"}, "embeddings": {0: "batch_size"}}),
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
        self.task = task

        # self.batch_size = batch_size
        # self.sequence_length = sequence_length
        # self.num_choices = num_choices
        # self.width = width
        # self.height = height
        # self.num_channels = num_channels
        # self.feature_size = feature_size
        # self.nb_max_frames = nb_max_frames

        self._axes: Dict[str, int] = {}

        self._validate_and_update_mandatory_axes(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            width=width,
            height=height,
            num_channels=num_channels,
            feature_size=feature_size,
            nb_max_frames=nb_max_frames,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.MANDATORY_AXES:
            if value is None:
                raise ValueError(f"Cannot set `None` to {name} because it is a mandatory axis.")
            self._axes[name] = value
        return super().__setattr__(name, value)

    def _validate_and_update_mandatory_axes(self, **kwargs):
        for axis in self.MANDATORY_AXES:
            if isinstance(axis, tuple):
                tasks, name = axis
                if not isinstance(tasks, tuple):
                    tasks = (tasks,)
                if self.task not in tasks:
                    continue
            else:
                name = axis
            axis_dim = kwargs[name]
            if axis_dim is None:
                if self._normalized_config.has_attribute(name):
                    axis_dim = getattr(self._normalized_config, name)
                else:
                    raise MissingMandatoryAxisDimension(
                        f"The value for the {name} axis is missing, it is needed to perform the export to TensorFlow Lite."
                    )
            self._axes[name] = axis_dim

    def _create_dummy_input_generator_classes(self) -> List["DummyInputGenerator"]:
        return [cls_(self.task, self._normalized_config, **self._axes) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES]

    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        raise NotImplementedError()

    @property
    def outputs(self) -> List[str]:
        return list(self._TASK_TO_COMMON_OUTPUTS[self.task].keys())

    def generate_dummy_inputs(self) -> Dict[str, "tf.Tensor"]:
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
                    f'Could not generate dummy inputs for "{input_name}". Try adding a proper dummy input generator to the model TFLite config.'
                )

        return dummy_inputs

    @property
    def inputs_specs(self) -> List["TensorSpec"]:
        dummy_inputs = self.generate_dummy_inputs()
        return [
            tf.TensorSpec(dummy_input.shape, dtype=dummy_input.dtype, name=input_name)
            for input_name, dummy_input in dummy_inputs.items()
        ]

    def model_to_tf_function(self, model: "TFPreTrainedModel", concrete: bool = False, **model_kwargs: Any):
        def forward(*args):
            input_names = self.inputs
            if len(args) != len(input_names):
                raise ArgumentError(
                    f"The number of inputs provided ({len(args)} do not match the number of expected inputs: :"
                    "{', '.join(input_names)}."
                )
            kwargs = dict(zip(input_names, args))
            outputs = model.call(**kwargs, **model_kwargs)
            return {key: value for key, value in outputs.items() if key in self.outputs}

        if concrete:
            function = tf.function(forward, input_signature=self.inputs_specs).get_concrete_function()
        else:
            function = tf.function(forward)

        return function
