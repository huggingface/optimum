#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
import dataclasses
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from packaging import version
from transformers.onnx.utils import (
    ParameterFormat,
    compute_effective_axis_dimension,
    compute_serialized_parameters_size,
)
from transformers.utils import TensorType, is_torch_available, is_vision_available, logging


if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import torch
from transformers.onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast


logger = logging.get_logger(__name__)


class OnnxConfigWithLoss(OnnxConfig, ABC):
    """
    Wrapper for the childern classes of `transformers.onnx.OnnxConfig` to export the model through the ONNX format with loss in outputs.
    """

    _tasks_to_extra_inputs = {
        "default": OrderedDict({"labels": {0: "batch"}}),
        "masked-lm": OrderedDict({"labels": {0: "batch", 1: "sequence"}}),
        "causal-lm": OrderedDict({"labels": {0: "batch", 1: "sequence"}}),
        "seq2seq-lm": OrderedDict({"labels": {0: "batch", 1: "sequence"}}),
        "sequence-classification": OrderedDict({"labels": {0: "batch"}}),
        "token-classification": OrderedDict({"labels": {0: "batch", 1: "sequence"}}),
        "multiple-choice": OrderedDict({"labels": {0: "batch"}}),
        "question-answering": OrderedDict(
            {
                "start_positions": {0: "batch"},
                "end_positions": {0: "batch"},
            }
        ),
        "image-classification": OrderedDict({"labels": {0: "batch"}}),
    }
    _tasks_to_extra_outputs = {
        "default": OrderedDict({"loss": {}}),
    }

    def __init__(self, config: OnnxConfig):
        self.__dict__ = config.__dict__
        self._onnx_config = config
        if self.task not in self._tasks_to_extra_inputs:
            raise ValueError(
                f"{self.task} is not a supported task, supported tasks: {self._tasks_to_extra_inputs.keys()}"
            )

    @classmethod
    def from_model_config(cls, config: OnnxConfig) -> "OnnxConfigWithLoss":
        """
        Instantiate a OnnxConfigWithLoss for a specific model
        Args:
            config: The model's configuration to use when exporting to ONNX
        Returns:
            OnnxConfigWithLoss for this model
        """
        return cls(config)

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors to provide to the model
        Returns:
            For each input: its name associated to the axes symbolic name and the axis position within the tensor
        """
        inputs = self._onnx_config.inputs
        inputs.update(self._tasks_to_extra_inputs[self.task])
        return inputs

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors to provide to the model
        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        common_outputs = self._tasks_to_common_outputs[self.task]
        common_outputs.update(self._tasks_to_extra_outputs["default"])
        if "loss" in common_outputs.keys():
            common_outputs.move_to_end("loss", last=False)
        return copy.deepcopy(common_outputs)

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        tokenizer: "PreTrainedTokenizerBase" = None,
    ) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework
        Args:
            preprocessor: ([`PreTrainedTokenizerBase`] or [`FeatureExtractionMixin`]):
                The preprocessor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2)
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.
        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """
        # Generate dummy labels
        dummy_inputs = super().generate_dummy_inputs(
            preprocessor,
            batch_size,
            seq_length,
            is_pair,
            framework,
            num_channels,
            image_width,
            image_height,
            tokenizer,
        )
        for label, input in self._tasks_to_extra_inputs[self.task].items():
            if "sequence" in input.keys():
                dummy_inputs[label] = torch.zeros(
                    self.default_fixed_batch, self.default_fixed_sequence, dtype=torch.long
                )
            else:
                dummy_inputs[label] = torch.zeros(self.default_fixed_batch, dtype=torch.long)
        return dummy_inputs


class OnnxConfigWithPastAndLoss(OnnxConfigWithLoss, ABC):
    def __init__(
        self,
        config: OnnxConfigWithPast,
        use_past: bool = False,
    ):
        super().__init__(config)
        self.use_past = use_past

    @classmethod
    def with_past(cls, config: OnnxConfigWithPast) -> "OnnxConfigWithPast":
        """
        Instantiate a OnnxConfigWithPast with `use_past` attribute set to True
        Args:
            config: The underlying model's config to use when exporting to ONNX
        Returns:
            OnnxConfigWithPast with `.use_past = True`
        """
        return cls(config, use_past=True)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super().outputs
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}

        return None

    @property
    def num_layers(self) -> int:
        """
        The number of layers attribute retrieved from the model config. Override this for model configs where the
        number of layers attribute is not called `num_layers`.
        """
        if not hasattr(self._onnx_config, "num_layers"):
            raise AttributeError(
                "could not find the number of layers attribute in the model configuration, override the num_layers property of the model OnnxConfig to solve this"
            )
        return self._onnx_config.num_layers

    @property
    def num_attention_heads(self) -> int:
        """
        The number of attention heads attribute retrieved from the model config. Override this for model configs where
        the number of attention heads attribute is not called `num_attention_heads`.
        """
        if not hasattr(self._onnx_config, "num_attention_heads"):
            raise AttributeError(
                "could not find the number of attention heads attribute in the model configuration, override the num_attention_heads property of the model OnnxConfig to solve this"
            )
        return self._onnx_config.num_attention_heads

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:

        dummy_inputs = self._onnx_config.generate_dummy_inputs(
            tokenizer,
            batch_size,
            seq_length,
            is_pair,
            framework,
        )
        for label, input in self._tasks_to_extra_inputs[self.task].items():
            if "sequence" in input.keys():
                dummy_inputs[label] = torch.zeros(
                    self.default_fixed_batch, self.default_fixed_sequence, dtype=torch.long
                )
            else:
                dummy_inputs[label] = torch.zeros(self.default_fixed_batch, dtype=torch.long)
        return dummy_inputs

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.key"] = t[0]
        flattened_output[f"{name}.{idx}.value"] = t[1]

    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        flattened_output = {}
        if name in ["present", "past_key_values"]:
            for idx, t in enumerate(field):
                self._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            flattened_output = super().flatten_output_collection_property(name, field)

        return flattened_output


class OnnxSeq2SeqConfigWithPastAndLoss(OnnxSeq2SeqConfigWithPast, ABC):
    pass
