# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import copy
import dataclasses
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from packaging import version

from transformers.utils import is_torch_available, is_vision_available, logging
from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size


if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig


if is_vision_available():
    from PIL import Image

logger = logging.get_logger(__name__)


# 2 Gb
EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024


@dataclasses.dataclass
class PatchingSpec:
    """
    Data class that holds patching specifications.

    Args:
        o: Module / object where the op to patch is located
        name: Name of the op to monkey patch
        custom_op: Custom op that patches the original op
        orig_op: Original op that is being patched
        op_wrapper: Wrapper (optional) that wraps both the original and custom ops.
            It is useful for ops that are class or static methods for instance.
    """

    o: Any
    name: str
    custom_op: Callable
    orig_op: Optional[Callable] = None
    op_wrapper: Optional[Callable] = None


class OnnxConfig(ABC):
    """
    Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.
    """
    NORMALIZED_CONFIG_CLASS = None
    DUMMY_INPUT_GENERATOR_CLASSES = ()
    DEFAULT_ONNX_OPSET = 11
    torch_onnx_minimum_version = version.parse("1.8")
    _tasks_to_common_outputs = {
        "causal-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "default": OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}),
        "image-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "image-segmentation": OrderedDict(
            {
                "logits": {0: "batch", 1: "sequence"},
                "pred_boxes": {0: "batch", 1: "sequence"},
                "pred_masks": {0: "batch", 1: "sequence"},
            }
        ),
        "masked-im": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "masked-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "multiple-choice": OrderedDict({"logits": {0: "batch"}}),
        "object-detection": OrderedDict(
            {
                "logits": {0: "batch", 1: "sequence"},
                "pred_boxes": {0: "batch", 1: "sequence"},
            }
        ),
        "question-answering": OrderedDict(
            {
                "start_logits": {0: "batch", 1: "sequence"},
                "end_logits": {0: "batch", 1: "sequence"},
            }
        ),
        "semantic-segmentation": OrderedDict({"logits": {0: "batch", 1: "num_labels", 2: "height", 3: "width"}}),
        "seq2seq-lm": OrderedDict({"logits": {0: "batch", 1: "decoder_sequence"}}),
        "sequence-classification": OrderedDict({"logits": {0: "batch"}}),
        "token-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
    }

    def __init__(self, config: "PretrainedConfig", task: str = "default", patching_specs: Optional[List[PatchingSpec]] = None):
        self._config = config
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)

        if task not in self._tasks_to_common_outputs:
            raise ValueError(
                f"{task} is not a supported task, supported tasks: {self._tasks_to_common_outputs.keys()}"
            )
        self.task = task

        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

    @property
    @abstractmethod
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
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        return 1e-5

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

    @staticmethod
    def use_external_data_format(num_parameters: int) -> bool:
        """
        Flag indicating if the model requires using external data format

        Args:
            num_parameters: Number of parameter on the model

        Returns:
            True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise
        """

        return (
            compute_serialized_parameters_size(num_parameters, ParameterFormat.Float)
            >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT
        )

    # TODO: make it possible to pass static shapes (batch size, sequence length, num choices, image width / height / channel)
    def generate_dummy_inputs(self, framework: str = "pt") -> Mapping[str, "torch.Tensor"]:
        generators = [cls_(self.task, self._normalized_config) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES]
        dummy_inputs = OrderedDict()
        for input_name in self.inputs:
            input_was_inserted = False
            for dummy_input_gen in generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework=framework)
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(f"Could not generate dummy input for \"{input_name}\", try adding the proper dummy input generator to the model onnx config.")
        return dummy_inputs

    # def _generate_dummy_images(
    #     self, batch_size: int = 2, num_channels: int = 3, image_height: int = 40, image_width: int = 40
    # ):
    #     images = []
    #     for _ in range(batch_size):
    #         data = np.random.rand(image_height, image_width, num_channels) * 255
    #         images.append(Image.fromarray(data.astype("uint8")).convert("RGB"))
    #     return images

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
