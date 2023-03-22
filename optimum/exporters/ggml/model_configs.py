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

from typing import Dict, List

import torch

from ...utils import NormalizedTextConfig
from .base import GgmlConfig, TextGgmlConfig


class BloomGgmlConfig(TextGgmlConfig):
    MODULE_MAP = {
        "word_embeddings": "tok_embeddings",
        "word_embeddings_layernorm": "norm",
        "input_layernorm": "attention_norm",
        "self_attention.query_key_value": "attention.query_key_value",
        "self_attention.dense": "attention.wo",
        "post_attention_layernorm": "ffn_norm",
        "mlp.dense_h_to_4h": "feed_forward.w1",
        "mlp.dense_4h_to_h": "feed_forward.w2",
        "ln_f": "output_norm",
        "lm_head": "output",
    }

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")

    def patch_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for name, param in state_dict.items():
            if "query_key_value" in name:
                q, k, v = param.reshape(self._normalized_config.num_attention_heads, 3, -1).unbind(1)
                state_dict[name] = torch.cat([q, k, v], dim=0).reshape_as(param)

        return state_dict

    @property
    def header_data(self):
        # reference: https://github.com/NouamaneTazi/bloomz.cpp/blob/main/convert-hf-to-ggml.py
        return [
            self._normalized_config.vocab_size,
            self._normalized_config.hidden_size,
            1,  # multiple_of, what the hell is this?
            self._normalized_config.num_attention_heads,
            self._normalized_config.num_layers,
        ]

    def get_name_map(self, parameters_names: List[str]) -> Dict[str, str]:
        name_map = {}
        for name in parameters_names:
            src = name
            nn = name
            if name != "lm_head.weight":
                nn = nn.split(".")[1:]
            else:
                nn = nn.split(".")

            if nn[0] == "h":
                nn[0] = "layers"
                mapped = self.MODULE_MAP[".".join(nn[2:-1])]
                name = ".".join(nn[:2] + [mapped] + nn[-1:])
            else:
                mapped = self.MODULE_MAP[".".join(nn[:-1])]
                name = ".".join([mapped] + nn[-1:])

            name_map[src] = name

        return name_map


class WhisperGgmlConfig(GgmlConfig):
    pass


class GPT2GgmlConfig(TextGgmlConfig):
    pass


class GPTJGgmlConfig(TextGgmlConfig):
    @property
    def header_data(self):
        # reference: https://github.com/ggerganov/ggml/blob/master/examples/gpt-j/convert-h5-to-ggml.py
        header_data = super().header_data()
        header_data.append(self._normalized_config.rotary_dim)
