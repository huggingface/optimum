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
import dataclasses
import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ...utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel

    from .base import OnnxConfig

logger = logging.get_logger(__name__)


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


class ModelPatcher:
    def __init__(self, config: "OnnxConfig", model: Union["PreTrainedModel", "TFPreTrainedModel"]):
        self._model = model

        patching_specs = config.PATCHING_SPECS
        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

        self.orig_forward_name = "forward" if hasattr(self._model, "forward") else "call"
        self.orig_forward = getattr(self._model, self.orig_forward_name)

        # TODO: remove that once we got rid of OnnxConfigWithLoss or we implemented it better.
        if config.__class__.__name__ == "OnnxConfigWithLoss":
            self.real_config = config._onnx_config
        else:
            self.real_config = config
        allow_past_in_outputs = (
            hasattr(self.real_config, "use_present_in_outputs") and self.real_config.use_present_in_outputs
        )

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            outputs = self.orig_forward(*args, **kwargs)

            filterd_outputs = {}
            for k, v in outputs.items():
                if config.torch_to_onnx_output_map.get(k, k) in config.outputs or (
                    allow_past_in_outputs and k.startswith("past_key_values")
                ):
                    filterd_outputs[k] = v
            return filterd_outputs

        self.patched_forward = patched_forward

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)

    def restore_ops(self):
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)

    def __enter__(self):
        self.patch_ops()
        setattr(self._model, self.orig_forward_name, self.patched_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)

    def __call__(self, *args, **kwargs):
        if getattr(self._model, self.orig_forward_name) is self.orig_forward:
            logger.warning("Running the non-patched model")
        return self._model(*args, **kwargs)


class Seq2SeqModelPatcher(ModelPatcher):
    def __init__(self, config: "OnnxConfig", model: Union["PreTrainedModel", "TFPreTrainedModel"]):
        super().__init__(config, model)

        allow_past_in_outputs = (
            hasattr(self.real_config, "use_present_in_outputs") and self.real_config.use_present_in_outputs
        )

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            outputs = self.orig_forward(*args, **kwargs)

            # Filter out cross attention past key values
            filterd_outputs = {}
            for name, value in outputs.items():
                if name != "past_key_values":
                    if self.real_config._behavior == "decoder" and name == "encoder_last_hidden_state":
                        # who cares about the encoder outputs in the decoder?
                        continue
                    else:
                        filterd_outputs[name] = value

                if config.torch_to_onnx_output_map.get(name, name) in config.outputs or (
                    allow_past_in_outputs and name.startswith("past_key_values")
                ):
                    if name == "past_key_values":
                        if self.real_config._behavior == "monolith" or (
                            self.real_config._behavior == "decoder" and self.real_config.use_past is False
                        ):
                            filterd_outputs[name] = value
                        elif self.real_config._behavior == "decoder" and self.real_config.use_past is True:
                            filterd_outputs[name] = tuple([v[:2] for v in value])
            return filterd_outputs

        self.patched_forward = patched_forward


class WavLMModelPatcher(ModelPatcher):
    def __init__(self, config: "OnnxConfig", model: Union["PreTrainedModel", "TFPreTrainedModel"]):
        super().__init__(config, model)

        allow_past_in_outputs = (
            hasattr(self.real_config, "use_present_in_outputs") and self.real_config.use_present_in_outputs
        )

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            args = list(args)

            signature = inspect.signature(self.orig_forward)
            output_attentions_index = list(signature.parameters.keys()).index("output_attentions")

            # setting output_attentions=True in the model input to avoid calling torch.nn.functional.scaled_dot_product_attention
            # in https://github.com/huggingface/transformers/blob/v4.27.1/src/transformers/models/wavlm/modeling_wavlm.py#L496
            # that calls https://github.com/pytorch/pytorch/blob/v2.0.0/torch/nn/functional.py#L5334
            args[output_attentions_index] = True
            outputs = self.orig_forward(*args, **kwargs)

            filterd_outputs = {}
            for k, v in outputs.items():
                if config.torch_to_onnx_output_map.get(k, k) in config.outputs or (
                    allow_past_in_outputs and k.startswith("past_key_values")
                ):
                    filterd_outputs[k] = v
            return filterd_outputs

        self.patched_forward = patched_forward
