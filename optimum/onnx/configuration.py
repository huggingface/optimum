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

from transformers.onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast


logger = logging.get_logger(__name__)


class OnnxConfigWithLoss(OnnxConfig, ABC):
    """
    Wrapper for the childern classes of `transformers.onnx.OnnxConfig` to export the model through the ONNX format with loss in outputs.
    """

    default_fixed_batch = 2
    default_fixed_sequence = 8
    torch_onnx_minimum_version = version.parse("1.8")
    _tasks_to_common_inputs_to_add = {
        "default": ("labels", {0: "batch"}),
    }
    _tasks_to_common_outputs_with_loss = {
        "default": OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}),
        "masked-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "causal-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "seq2seq-lm": OrderedDict({"logits": {0: "batch", 1: "decoder_sequence"}}),
        "sequence-classification": OrderedDict({"logits": {0: "batch"}}),
        "token-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "multiple-choice": OrderedDict({"logits": {0: "batch"}}),
        "question-answering": OrderedDict(
            {
                "start_logits": {0: "batch", 1: "sequence"},
                "end_logits": {0: "batch", 1: "sequence"},
            }
        ),
        "image-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
    }

    def __init__(self, config: OnnxConfig):
        self._config = config

        if self._config.task not in self._tasks_to_common_outputs_with_loss:
            raise ValueError(
                f"{self._config.task} is not a supported task, supported tasks: {self._tasks_to_common_outputs_with_loss.keys()}"
            )
        self.task = self._config.task

        self._patching_specs = []
        for spec in self._config.patching_specs if self._config.patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

    @classmethod
    def from_model_config(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfig":
        """
        Instantiate a OnnxConfig for a specific model
        Args:
            config: The model's configuration to use when exporting to ONNX
        Returns:
            OnnxConfig for this model
        """
        return cls(config, task=task)

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors to provide to the model
        Returns:
            For each input: its name associated to the axes symbolic name and the axis position within the tensor
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors to provide to the model
        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        common_outputs = self._tasks_to_common_outputs[self.task]
        return copy.deepcopy(common_outputs)

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting
        Returns:
            Dictionary with the keys (and their corresponding values) to override
        """
        if hasattr(self._config, "use_cache"):
            return {"use_cache": False}

        return None

    @property
    def is_torch_support_available(self) -> bool:
        """
        The minimum PyTorch version required to export the model.
        Returns:
            `bool`: Whether the installed version of PyTorch is compatible with the model.
        """
        if is_torch_available():
            from transformers.utils import torch_version

            return torch_version >= self.torch_onnx_minimum_version
        else:
            return False

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
            return dict(preprocessor(dummy_input, return_tensors=framework))
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            return dict(preprocessor(images=dummy_input, return_tensors=framework))
        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)

    def restore_ops(self):
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)

    @classmethod
    def flatten_output_collection_property(cls, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        """
        Flatten any potential nested structure expanding the name of the field with the index of the element within the
        structure.
        Args:
            name: The name of the nested structure
            field: The structure to, potentially, be flattened
        Returns:
            (Dict[str, Any]): Outputs with flattened structure and key mapping this new structure.
        """
        from itertools import chain

        return {f"{name}.{idx}": item for idx, item in enumerate(chain.from_iterable(field))}


class OnnxConfigWithPastAndLoss(OnnxConfigWithPast, ABC):
    pass


class OnnxSeq2SeqConfigWithPastAndLoss(OnnxSeq2SeqConfigWithPast, ABC):
    pass
