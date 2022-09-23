# coding=utf-8
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
from collections import OrderedDict
from typing import Any, Mapping, Optional

from ...utils import NormalizedConfig
from .config import AutoEncoderOnnxConfig, DecoderOnnxConfig


class BertOnnxConfig(AutoEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )


class DistilBertOnnxConfig(BertOnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )


class RobertaOnnxConfig(DistilBertOnnxConfig):
    pass


class GPT2NormalizedConfig(NormalizedConfig):
    NUM_LAYERS = "n_layer"
    NUM_ATTENTION_HEADS = "n_head"


class GPT2OnnxConfig(DecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = GPT2NormalizedConfig

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.add_past_key_values(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    # @property
    # def values_override(self) -> Optional[Mapping[str, Any]]:
    #     if not getattr(self._config, "pad_token_id", None):
    #         return {"pad_token_id": 0}


class GPTNeoNormalizedConfig(NormalizedConfig):
    NUM_ATTENTION_HEADS = "num_heads"


class GPTNeoOnnxConfig(DecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = GPTNeoNormalizedConfig

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.add_past_key_values(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    # def generate_dummy_inputs(
    #     self,
    #     tokenizer: PreTrainedTokenizer,
    #     batch_size: int = -1,
    #     seq_length: int = -1,
    #     is_pair: bool = False,
    #     framework: Optional[TensorType] = None,
    # ) -> Mapping[str, Any]:

    #     common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
    #         tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
    #     )

    #     # We need to order the input in the way they appears in the forward()
    #     ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

    #     # Need to add the past_keys
    #     if self.use_past:
    #         if not is_torch_available():
    #             raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
    #         else:
    #             import torch

    #             batch, seqlen = common_inputs["input_ids"].shape
    #             # Not using the same length for past_key_values
    #             past_key_values_length = seqlen + 2
    #             past_shape = (
    #                 batch,
    #                 self.num_attention_heads,
    #                 past_key_values_length,
    #                 self._config.hidden_size // self.num_attention_heads,
    #             )
    #             ordered_inputs["past_key_values"] = [
    #                 (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
    #             ]

    #     ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
    #     if self.use_past:
    #         mask_dtype = ordered_inputs["attention_mask"].dtype
    #         ordered_inputs["attention_mask"] = torch.cat(
    #             [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
    #         )

    #     return ordered_inputs
