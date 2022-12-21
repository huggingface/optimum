# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""ONNX configuration base classes."""

import copy
import dataclasses
import enum
import inspect
import itertools
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

from transformers.utils import is_torch_available

from ...utils import DEFAULT_DUMMY_SHAPES
from ...utils import TORCH_MINIMUM_VERSION as GLOBAL_MIN_TORCH_VERSION
from ...utils import DummyInputGenerator, DummyTrainingLabelsInputGenerator, logging
from ...utils.doc import add_dynamic_docstring
from ..base import ExportConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, TFPreTrainedModel

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


GENERATE_DUMMY_DOCSTRING = r"""
        Generates the dummy inputs necessary for tracing the model. If not explicitely specified, default input shapes are used.

        Args:
            framework (`str`, defaults to `"pt"`):
                The framework for which to create the dummy inputs.
            batch_size (`int`, defaults to {batch_size}):
                The batch size to use in the dummy inputs.
            sequence_length (`int`, defaults to {sequence_length}):
                The sequence length to use in the dummy inputs.
            num_choices (`int`, defaults to {num_choices}):
                The number of candidate answers provided for multiple choice task.
            image_width (`int`, defaults to {width}):
                The width to use in the dummy inputs for vision tasks.
            image_height (`int`, defaults to {height}):
                The height to use in the dummy inputs for vision tasks.
            num_channels (`int`, defaults to {num_channels}):
                The number of channels to use in the dummpy inputs for vision tasks.
            feature_size (`int`, defaults to {feature_size}):
                The number of features to use in the dummpy inputs for audio tasks in case it is not raw audio.
                This is for example the number of STFT bins or MEL bins.
            nb_max_frames (`int`, defaults to {nb_max_frames}):
                The number of frames to use in the dummpy inputs for audio tasks in case the input is not raw audio.
            audio_sequence_length (`int`, defaults to {audio_sequence_length}):
                The number of frames to use in the dummpy inputs for audio tasks in case the input is raw audio.

        Returns:
            `Dict`: A dictionary mapping the input names to dummy tensors in the proper framework format.
"""


class OnnxConfig(ExportConfig, ABC):
    """
    Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.

    Class attributes:

    - NORMALIZED_CONFIG_CLASS (`Type`) -- A class derived from [`~optimum.utils.NormalizedConfig`] specifying how to
    normalize the model config.
    - DUMMY_INPUT_GENERATOR_CLASSES (`Tuple[Type]`) -- A tuple of classes derived from
    [`~optimum.utils.DummyInputGenerator`] specifying how to create dummy inputs.
    - ATOL_FOR_VALIDATION (`Union[float, Dict[str, float]]`) -- A float or a dictionary mapping task names to float,
    where the float values represent the absolute tolerance value to use during model conversion validation.
    - DEFAULT_ONNX_OPSET (`int`, defaults to 11) -- The default ONNX opset to use for the ONNX export.
    - MIN_TORCH_VERSION (`packaging.version.Version`, defaults to [`~optimum.exporters.onnx.utils.TORCH_MINIMUM_VERSION`]) -- The
    minimum torch version supporting the export of the model to ONNX.

    Args:
        config (`transformers.PretrainedConfig`):
            The model configuration.
        task (`str`, defaults to `"default"`):
            The task the model should be exported for.
    """

    NORMALIZED_CONFIG_CLASS = None
    DUMMY_INPUT_GENERATOR_CLASSES = ()
    DEFAULT_ONNX_OPSET = 11
    ATOL_FOR_VALIDATION: Union[float, Dict[str, float]] = 1e-5
    MIN_TORCH_VERSION = GLOBAL_MIN_TORCH_VERSION
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
        "speech2seq-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
    }

    def __init__(
        self, config: "PretrainedConfig", task: str = "default", patching_specs: Optional[List[PatchingSpec]] = None
    ):
        if task not in self._TASK_TO_COMMON_OUTPUTS:
            raise ValueError(
                f"{task} is not a supported task, supported tasks: {', '.join(self._TASK_TO_COMMON_OUTPUTS.keys())}"
            )
        self.task = task

        self._config = config
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)

        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

    def _create_dummy_input_generator_classes(self, **kwargs) -> List[DummyInputGenerator]:
        """
        Instantiates the dummy input generators from `self.DUMMY_INPUT_GENERATOR_CLASSES`.
        Each dummy input generator is independent, so this method instantiates the first generator, and
        forces the other generators to use the same batch size, meaning they will all produce inputs of the same batch
        size. Override this method for custom behavior.
        """
        first_inputs_gen = self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config, **kwargs)
        dummy_inputs_generators = [
            cls_(self.task, self._normalized_config, **kwargs) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES[1:]
        ]
        dummy_inputs_generators.insert(0, first_inputs_gen)

        return dummy_inputs_generators

    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors to provide to the model.

        Returns:
            `Mapping[str, Mapping[int, str]]`: A mapping of each input name to a mapping of axis position to the axes symbolic name.
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors to provide to the model.

        Returns:
            `Mapping[str, Mapping[int, str]]`: A mapping of each output name to a mapping of axis position to the axes symbolic name.
        """
        common_outputs = self._TASK_TO_COMMON_OUTPUTS[self.task]
        return copy.deepcopy(common_outputs)

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting.

        Returns:
            `Optional[Mapping[str, Any]]`: A dictionary specifying the configuration items to override.
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
            from ...utils import torch_version

            return torch_version >= self.MIN_TORCH_VERSION
        return False

    @property
    def torch_to_onnx_input_map(self) -> Mapping[str, str]:
        """
        Dictionary of keys to update the ONNX input name for export. Override the function when
        the dummy input names and the exported ONNX input names need to be different.

        Returns:
            `Mapping[str, str]`: A dictionary specifying the dummy input name to exported ONNX input name map.
        """
        return {}

    def ordered_inputs(self, model: Union["PreTrainedModel", "TFPreTrainedModel"]) -> Mapping[str, Mapping[int, str]]:
        """
        Re-orders the inputs using the model forward pass signature.

        Args:
            model ([`transformers.PreTrainedModel`] or [`transformers.TFPreTrainedModel`]):
                The model for which we will use the OnnxConfig.

        Returns:
            `Mapping[str, Mappingp[int, str]]`: The properly ordered inputs.
        """
        inputs = self.inputs

        ordered_inputs = {}
        if hasattr(model, "forward"):
            sig = inspect.signature(model.forward)
        else:
            sig = inspect.signature(model.call)

        for param in sig.parameters:
            param_regex = re.compile(rf"{param}(\.\d*)?")
            to_insert = []
            for name, dynamic_axes in inputs.items():
                if re.match(param_regex, name):
                    to_insert.append((name, dynamic_axes))
            # TODO: figure out a smart way of re-ordering potential nested structures.
            # to_insert = sorted(to_insert, key=lambda t: t[0])
            for name, dynamic_axes in to_insert:
                name = self.torch_to_onnx_input_map.get(name, name)
                ordered_inputs[name] = dynamic_axes
        return ordered_inputs

    @add_dynamic_docstring(text=GENERATE_DUMMY_DOCSTRING, dynamic_elements=DEFAULT_DUMMY_SHAPES)
    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> Dict:
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        for input_name in self.inputs:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework=framework)
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                )
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
        Flattens any potential nested structure expanding the name of the field with the index of the element within the
        structure.

        Args:
            name (`str`):
                The name of the nested structure.
            field (`Iterable[Any]`):
                The structure to potentially flattened.

        Returns:
            `Dict[str, Any]`: Outputs with flattened structure and key mapping this new structure.

        """
        return {f"{name}.{idx}": item for idx, item in enumerate(itertools.chain.from_iterable(field))}

    def generate_dummy_inputs_for_validation(self, reference_model_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Generates inputs for ONNX Runtime using the reference model inputs. Override this to run inference with seq2seq
        models which have the encoder and decoder exported as separate ONNX files.
        Args:
            reference_model_inputs ([`Mapping[str, Tensor]`):
                Reference inputs for the model.
        Returns:
            `Mapping[str, Tensor]`: The mapping holding the kwargs to provide to the model's forward function
        """
        return reference_model_inputs

    def output_names_for_validation(self, reference_output_names: List[str]) -> List[str]:
        """
        Returns the output names of the reference model corresponding to the output names of the ONNX model.
        Useful to compare the outputs from the ONNX and the reference model when their output names differ.
        Args:
            reference_output_names ([`List[str]`):
                The original ONNX model output names.
        Returns:
            `List[str]`: The corresponding reference model output names.
        """
        return reference_output_names


class OnnxConfigWithPast(OnnxConfig, ABC):
    """
    Inherits from [`~exporters.onnx.OnnxConfig`]. A base class to handle the ONNX configuration of decoder-only models.
    """

    PAD_ATTENTION_MASK_TO_MATCH_TOTAL_SEQUENCE_LENGTH: bool = True
    USE_PAST_IN_INPUTS: Optional[bool] = None
    USE_PRESENT_IN_OUTPUTS: Optional[bool] = None

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "default",
        patching_specs: Optional[List[PatchingSpec]] = None,
        use_past: bool = False,
        use_past_in_inputs: Optional[bool] = None,
        use_present_in_outputs: Optional[bool] = None,
    ):
        self.use_past = use_past
        if use_past_in_inputs is None:
            use_past_in_inputs = self.USE_PAST_IN_INPUTS
        if use_present_in_outputs is None:
            use_present_in_outputs = self.USE_PRESENT_IN_OUTPUTS
        self.use_past_in_inputs = use_past if use_past_in_inputs is None else use_past_in_inputs
        self.use_present_in_outputs = use_past if use_present_in_outputs is None else use_present_in_outputs
        if use_past != self.use_past_in_inputs:
            logger.warning(
                f"use_past = {use_past} is different than use_past_in_inputs = {use_past_in_inputs}, the value of "
                "use_past_in_inputs will used for the inputs."
            )
        if use_past != self.use_present_in_outputs:
            logger.warning(
                f"use_past = {use_past} is different than use_present_in_outputs = {use_present_in_outputs}, the value "
                "of use_present_in_outputs value will be used for the outputs."
            )
        super().__init__(config, task=task, patching_specs=patching_specs)

    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfigWithPast":
        """
        Instantiates a [`~optimum.exporters.onnx.OnnxConfig`] with `use_past` attribute set to `True`.

        Args:
            config (`transformers.PretrainedConfig`):
                The underlying model's config to use when exporting to ONNX.
            task (`str`, defaults to `"default"`):
                The task the model should be exported for.

        Returns:
            [`~optimum.exporters.onnx.OnnxConfig`]: The onnx config with `.use_past = True`
        """
        return cls(config, task=task, use_past=True)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super().outputs
        if self.use_present_in_outputs:
            self.add_past_key_values(common_outputs, direction="outputs")
        return common_outputs

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past_in_inputs or self.use_present_in_outputs}

    @add_dynamic_docstring(text=GENERATE_DUMMY_DOCSTRING, dynamic_elements=DEFAULT_DUMMY_SHAPES)
    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = [key for key in self.inputs.keys() if not key.startswith("past_key_values")]
        if self.use_past:
            input_names.append("past_key_values")
        for input_name in input_names:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    if self.use_past is True and input_name == "decoder_input_ids":
                        sequence_length = dummy_input_gen.sequence_length
                        if "sequence_length" in kwargs and kwargs["sequence_length"] != 1:
                            logger.info(
                                f"Asked a sequence length of {kwargs['sequence_length']}, but a sequence length of 1 will be used with use_past ==True for `decoder_input_ids`."
                            )
                        dummy_input_gen.sequence_length = 1
                        dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework=framework)
                        dummy_input_gen.sequence_length = sequence_length
                    else:
                        dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework=framework)
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                )

        if (
            self.PAD_ATTENTION_MASK_TO_MATCH_TOTAL_SEQUENCE_LENGTH
            and self.use_past_in_inputs
            and "attention_mask" in dummy_inputs
        ):
            past_length = dummy_inputs["past_key_values"][0][0].shape[2]
            dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                dummy_inputs["attention_mask"],
                padding_length=past_length,
                dim=1,
                dtype=dummy_inputs["attention_mask"].dtype,
            )

        return dummy_inputs

    def add_past_key_values(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        """
        Fills `input_or_outputs` mapping with past_key_values dynamic axes considering the direction.

        Args:
            inputs_or_outputs (`Mapping[str, Mapping[int, str]]`):
                The mapping to fill.
            direction (`str`):
                either "inputs" or "outputs", it specifies whether `input_or_outputs` is the input mapping or the
                output mapping, this is important for axes naming.
        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        name = "past_key_values" if direction == "inputs" else "present"
        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 2: "past_sequence_length + sequence_length"}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 2: "past_sequence_length + sequence_length"}

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.key"] = t[0]
        flattened_output[f"{name}.{idx}.value"] = t[1]

    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        flattened_output = {}
        if name in ["present", "past_key_values"]:
            for idx, t in enumerate(field):
                self.flatten_past_key_values(flattened_output, name, idx, t)
        else:
            flattened_output = super().flatten_output_collection_property(name, field)

        return flattened_output


class ConfigBehavior(str, enum.Enum):
    """
    Specifies the behavior of the [`~exporters.onnx.base.OnnxSeq2SeqConfigWithPast`]:
        - MONOLITH: the config can be used to export the whole seq2seq model as a single file.
        - ENCODER: the config can be used to export the encoder part of the seq2seq model.
        - DECODER: the config can be used to export the decoder part of the seq2seq model.
    """

    MONOLITH = "monolith"
    ENCODER = "encoder"
    DECODER = "decoder"


class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):
    """
    Inherits from [`~exporters.onnx.OnnxConfigWithPast`]. A base class to handle the ONNX configuration of encoder-decoder models.
    """

    PAD_ATTENTION_MASK_TO_MATCH_TOTAL_SEQUENCE_LENGTH = False

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "default",
        patching_specs: Optional[List[PatchingSpec]] = None,
        use_past: bool = False,
        use_past_in_inputs: Optional[bool] = None,
        use_present_in_outputs: Optional[bool] = None,
        behavior: ConfigBehavior = ConfigBehavior.MONOLITH,
    ):
        super().__init__(
            config,
            task=task,
            patching_specs=patching_specs,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            use_present_in_outputs=use_present_in_outputs,
        )
        self._behavior = behavior
        self.override_attributes_for_behavior()

    def override_attributes_for_behavior(self):
        """Override this to specify custom attribute change for a given behavior."""
        if self._behavior is ConfigBehavior.ENCODER:
            self.task = "default"
            self.use_past_in_inputs = False
            self.use_present_in_outputs = False
        if self._behavior is ConfigBehavior.DECODER:
            self.use_past_in_inputs = self.use_past
            self.use_present_in_outputs = True

    def with_behavior(
        self, behavior: Union[str, ConfigBehavior], use_past: bool = False
    ) -> "OnnxSeq2SeqConfigWithPast":
        """
        Creates a copy of the current OnnxConfig but with a different `ConfigBehavior` and `use_past` value.

        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
            use_past (`bool`, defaults to `False`):
                Whether or not the new instance should use past.

        Returns:
            `OnnxSeq2SeqConfigWithPast`
        """
        if isinstance(behavior, str) and not isinstance(behavior, ConfigBehavior):
            behavior = ConfigBehavior(behavior)
        return self.__class__(
            self._config,
            task=self.task,
            patching_specs=self._patching_specs,
            use_past=use_past,
            behavior=behavior,
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super(OnnxConfigWithPast, self).outputs
        # Renaming the outputs axes properly.
        for name, axes_names in common_outputs.items():
            sequence_name = "encoder_sequence_length" if "encoder" in name else "decoder_sequence_length"
            for axis_idx, name in axes_names.items():
                if "sequence" in name:
                    axes_names[axis_idx] = sequence_name
                # We reset the value as the order in common_outputs (OrderedDict) is lost otherwise
                else:
                    axes_names[axis_idx] = name
        if self.use_present_in_outputs:
            self.add_past_key_values(common_outputs, direction="outputs")

        return common_outputs

    def add_past_key_values(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        name = "past_key_values" if direction == "inputs" else "present"
        encoder_sequence = "past_encoder_sequence_length"
        decoder_sequence = (
            "past_decoder_sequence_length"
            if direction == "inputs"
            else "past_decoder_sequence_length + sequence_length"
        )
        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch_size", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch_size", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch_size", 2: encoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch_size", 2: encoder_sequence}

        if direction == "outputs" and "encoder_last_hidden_state" in inputs_or_outputs:
            inputs_or_outputs.move_to_end("encoder_last_hidden_state")

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.decoder.key"] = t[0]
        flattened_output[f"{name}.{idx}.decoder.value"] = t[1]
        flattened_output[f"{name}.{idx}.encoder.key"] = t[2]
        flattened_output[f"{name}.{idx}.encoder.value"] = t[3]


class OnnxConfigWithLoss(OnnxConfig, ABC):
    """
    Wrapper for the children classes of `optimum.exporters.onnx.OnnxConfig` to export the model through the ONNX format
    with loss in outputs and labels in the inputs. For seq-to-seq models, labels will be appended to the inputs of
    decoders.
    """

    _tasks_to_extra_inputs = {
        "default": {"labels": {0: "batch_size"}},
        "masked-lm": {"labels": {0: "batch_size", 1: "sequence_length"}},
        "causal-lm": {"labels": {0: "batch_size", 1: "sequence_length"}},
        "seq2seq-lm": {"labels": {0: "batch_size", 1: "sequence_length"}},
        "sequence-classification": {"labels": {0: "batch_size"}},
        "token-classification": {"labels": {0: "batch_size", 1: "sequence_length"}},
        "multiple-choice": {"labels": {0: "batch_size"}},
        "question-answering": {
            "start_positions": {0: "batch_size"},
            "end_positions": {0: "batch_size"},
        },
        "image-classification": {"labels": {0: "batch_size"}},
        "seq2seq-lm": {"labels": {0: "batch_size", 1: "sequence_length"}},
    }
    _tasks_to_extra_outputs = {
        "default": OrderedDict({"loss": {}}),
    }

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTrainingLabelsInputGenerator,)

    def __init__(self, config: OnnxConfig):
        self._onnx_config = config
        self.task = self._onnx_config.task
        self._normalized_config = self._onnx_config._normalized_config
        self._patching_specs = self._onnx_config._patching_specs

    @classmethod
    def from_model_config(cls, config: OnnxConfig) -> "OnnxConfigWithLoss":
        return cls(config)

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        inputs = self._onnx_config.inputs
        inputs.update(self._tasks_to_extra_inputs[self.task])
        return inputs

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = self._onnx_config.outputs
        extra_outputs = self._tasks_to_extra_outputs["default"]
        common_outputs.update(extra_outputs)
        for key in reversed(extra_outputs.keys()):
            common_outputs.move_to_end(key, last=False)
        return copy.deepcopy(common_outputs)

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = self._onnx_config.generate_dummy_inputs(framework=framework, **kwargs)
        input_name, _ = next(iter(self._onnx_config.inputs.items()))
        batch_size = dummy_inputs[input_name].shape[0]

        if isinstance(self._onnx_config, OnnxSeq2SeqConfigWithPast) and self._onnx_config.use_past_in_inputs is True:
            kwargs["sequence_length"] = 1

        dummy_inputs_generators = [
            cls_(self.task, self._normalized_config, batch_size=batch_size, **kwargs)
            for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES
        ]

        for input_name in self._tasks_to_extra_inputs[self.task]:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework=framework)
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                )

        return dummy_inputs

    def generate_dummy_inputs_for_validation(self, reference_model_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)

    @property
    def torch_to_onnx_input_map(self) -> Mapping[str, str]:
        return self._onnx_config.torch_to_onnx_input_map

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        return self._onnx_config.values_override
