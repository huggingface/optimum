# Copyright 2022 The HuggingFace and Meta Team.  All rights reserved.
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
import torch
import torch.nn as nn

from ..base import BetterTransformerBaseLayer


class WhisperEncoderLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, whisper_layer, config):
        super().__init__(config)
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    whisper_layer.self_attn.q_proj.weight,
                    whisper_layer.self_attn.k_proj.weight,
                    whisper_layer.self_attn.v_proj.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    whisper_layer.self_attn.q_proj.bias,
                    torch.zeros_like(whisper_layer.self_attn.q_proj.bias),
                    whisper_layer.self_attn.v_proj.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = whisper_layer.self_attn.out_proj.weight
        self.out_proj_bias = whisper_layer.self_attn.out_proj.bias

        # Linear layer 1
        self.linear1_weight = whisper_layer.fc1.weight
        self.linear1_bias = whisper_layer.fc1.bias

        # Linear layer 2
        self.linear2_weight = whisper_layer.fc2.weight
        self.linear2_bias = whisper_layer.fc2.bias

        # Layer norm 1
        self.norm1_eps = whisper_layer.self_attn_layer_norm.eps
        self.norm1_weight = whisper_layer.self_attn_layer_norm.weight
        self.norm1_bias = whisper_layer.self_attn_layer_norm.bias

        # Layer norm 2
        self.norm2_eps = whisper_layer.final_layer_norm.eps
        self.norm2_weight = whisper_layer.final_layer_norm.weight
        self.norm2_bias = whisper_layer.final_layer_norm.bias

        # Model hyper parameters
        self.num_heads = whisper_layer.self_attn.num_heads
        self.embed_dim = whisper_layer.self_attn.embed_dim

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False
        self.norm_first = True

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        attention_mask = None  # attention mask seems to be always None

        hidden_states = torch._transformer_encoder_layer_fwd(
            hidden_states,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj_weight,
            self.out_proj_bias,
            self.use_gelu,
            self.norm_first,
            self.norm1_eps,
            self.norm1_weight,
            self.norm1_bias,
            self.norm2_weight,
            self.norm2_bias,
            self.linear1_weight,
            self.linear1_bias,
            self.linear2_weight,
            self.linear2_bias,
            attention_mask,
        )
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return (hidden_states,)
