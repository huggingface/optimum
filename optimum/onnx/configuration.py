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
        self._config = config._config
        self._onnx_config = config
        if self._onnx_config.task not in self._tasks_to_extra_inputs:
            raise ValueError(
                f"{self._onnx_config.task} is not a supported task, supported tasks: {self._tasks_to_extra_inputs.keys()}"
            )
        self.task = self._onnx_config.task
        self._patching_specs = self._onnx_config._patching_specs

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
        from transformers.feature_extraction_utils import FeatureExtractionMixin
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
            raise ValueError("You cannot provide both a tokenizer and a preprocessor to generate dummy inputs.")
        if tokenizer is not None:
            warnings.warn(
                "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use `preprocessor` instead.",
                FutureWarning,
            )
            logger.warning("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
            preprocessor = tokenizer
        if isinstance(preprocessor, PreTrainedTokenizerBase):
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(
                batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
            )
            # If dynamic axis (-1) we forward with a fixed dimension of 8 tokens to avoid optimizations made by ONNX
            token_to_add = preprocessor.num_special_tokens_to_add(is_pair)
            seq_length = compute_effective_axis_dimension(
                seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
            )
            # Generate dummy inputs according to compute batch and sequence
            dummy_input = [" ".join([preprocessor.unk_token]) * seq_length] * batch_size
            dummy_dict = dict(preprocessor(dummy_input, return_tensors=framework))
            # Generate dummy labels
            for label, input in self._tasks_to_extra_inputs[self.task].items():
                if "sequence" in input.keys():
                    dummy_dict[label] = torch.zeros(
                        self.default_fixed_batch, self.default_fixed_sequence, dtype=torch.long
                    )
                else:
                    dummy_dict[label] = torch.zeros(self.default_fixed_batch, dtype=torch.long)
            return dummy_dict
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            dummy_dict = dict(preprocessor(dummy_input, return_tensors=framework))
            # Generate dummy labels
            for label, input in self._tasks_to_extra_inputs[self.task].items():
                if "sequence" in input.keys():
                    dummy_dict[label] = torch.zeros(
                        self.default_fixed_batch, self.default_fixed_sequence, dtype=torch.long
                    )
                else:
                    dummy_dict[label] = torch.zeros(self.default_fixed_batch, dtype=torch.long)
            return dummy_dict
        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )


class OnnxConfigWithPastAndLoss(OnnxConfigWithPast, ABC):
    pass


class OnnxSeq2SeqConfigWithPastAndLoss(OnnxSeq2SeqConfigWithPast, ABC):
    pass
