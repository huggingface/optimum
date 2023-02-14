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
from typing import TYPE_CHECKING, Any, Dict, Optional

from transformers.file_utils import TensorType
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers.onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast


logger = logging.get_logger(__name__)


class EncoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}})


class DecoderOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "past_decoder_sequence + sequence"}),
                ("encoder_hidden_states", {0: "batch", 1: "encoder_sequence"}),
                ("encoder_attention_mask", {0: "batch", 1: "encoder_sequence"}),
            ]
        )
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        return common_inputs

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Dict[str, Any]:
        import torch

        common_inputs = {}
        dummy_input = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )
        batch, encoder_seq_length = dummy_input["input_ids"].shape
        encoder_hidden_states_shape = (batch, encoder_seq_length, self._config.hidden_size)
        common_inputs["input_ids"] = dummy_input.pop("decoder_input_ids")
        common_inputs["encoder_hidden_states"] = torch.zeros(encoder_hidden_states_shape)
        common_inputs["encoder_attention_mask"] = dummy_input.pop("attention_mask")

        if "past_key_values" in dummy_input:
            common_inputs["past_key_values"] = dummy_input.pop("past_key_values")

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super(OnnxConfigWithPast, self).outputs
        self.fill_with_past_key_values_(common_outputs, direction="outputs")
        return common_outputs

    def fill_with_past_key_values_(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        num_pkv_per_layer = 4
        _, num_decoder_layers = self.num_layers
        name = "past" if direction == "inputs" else "present"
        decoder_sequence = "past_decoder_sequence" if direction == "inputs" else "past_decoder_sequence + sequence"
        for i in range(num_decoder_layers * num_pkv_per_layer):
            inputs_or_outputs[f"{name}_key_values_{i}"] = {0: "batch", 2: decoder_sequence}
