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
    from ...utils import DummyInputGenerator


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
        if task not in self._tasks_to_common_outputs:
            raise ValueError(
                f"{task} is not a supported task, supported tasks: {self._tasks_to_common_outputs.keys()}"
            )
        self.task = task

        self._config = config
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)

        first_inputs_gen = self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config)
        self.dummy_inputs_generators = [cls_(self.task, self._normalized_config, batch_size=first_inputs_gen.batch_size) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES[1:]]
        self.dummy_inputs_generators.insert(0, first_inputs_gen)

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
        dummy_inputs = OrderedDict()
        for input_name in self.inputs:
            input_was_inserted = False
            for dummy_input_gen in self.dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework=framework)
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(f"Could not generate dummy input for \"{input_name}\", try adding the proper dummy input generator to the model onnx config.")
        return dummy_inputs

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


class OnnxConfigWithPast(OnnxConfig, ABC):
    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs)
        self.use_past = use_past

    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfigWithPast":
        """
        Instantiate a OnnxConfig with `use_past` attribute set to True

        Args:
            config: The underlying model's config to use when exporting to ONNX

        Returns:
            OnnxConfig with `.use_past = True`
        """
        return cls(config, task=task, use_past=True)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super().outputs
        if self.use_past:
            self.add_past_key_values(common_outputs, direction="outputs")

        return common_outputs

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}

    # TODO: make it possible to pass static shapes (batch size, sequence length, num choices, image width / height / channel)
    def generate_dummy_inputs(self, framework: str = "pt") -> Mapping[str, "torch.Tensor"]:
        dummy_inputs = super().generate_dummy_inputs(framework=framework)

        if self.use_past and "attention_mask" in dummy_inputs:
            import pdb; pdb.set_trace()
            batch_size, past_key_values_length = dummy_inputs["past_key_values.0.key"].shape
            mask_dtype = dummy_inputs["attention_mask"].dtype
            dummy_inputs["attention_mask"] = torch.cat(
                [dummy_inputs["attention_mask"], torch.ones(batch_size, past_key_values_length, dtype=mask_dtype)],
                dim=1,
            )
        return dummy_inputs

    def add_past_key_values(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        """
        Fill the input_or_outputs mapping with past_key_values dynamic axes considering the direction.

        Args:
            inputs_or_outputs: The mapping to fill.
            direction: either "inputs" or "outputs", it specifies whether input_or_outputs is the input mapping or the
                output mapping, this is important for axes naming.

        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        name = "past_key_values" if direction == "inputs" else "present"
        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}

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
