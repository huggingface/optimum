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


class AlbertLayerBetterTransformer(nn.Module):
    def __init__(self, albert_layer, config):
        r"""
        A simple conversion of the BERT layer to its `BetterTransformer` implementation.

        Args:
            albert_layer (`torch.nn.Module`):
                The original BERT Layer where the weights needs to be retrieved.
        """
        super().__init__()
        # Sanity checks
        self.act_fn = config.hidden_act
        self.norm_first = False
        if self.act_fn not in ["gelu", "relu", "gelu_new"]:
            raise ValueError(
                f"Activation function {self.act_fn} not supported" " for `BetterTransformer` integration."
            )
        if hasattr(config, "position_embedding_type") and config.position_embedding_type != "absolute":
            raise ValueError(
                f"Positional embedding type {config.position_embedding_type} not "
                "supported for `BetterTransformer` integration"
            )
        self.use_gelu = (self.act_fn == "gelu") or (self.act_fn == "gelu_new")

        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    albert_layer.attention.query.weight,
                    albert_layer.attention.key.weight,
                    albert_layer.attention.value.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    albert_layer.attention.query.bias,
                    albert_layer.attention.key.bias,
                    albert_layer.attention.value.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = albert_layer.attention.dense.weight
        self.out_proj_bias = albert_layer.attention.dense.bias

        # Linear layer 1
        self.linear1_weight = albert_layer.ffn.weight
        self.linear1_bias = albert_layer.ffn.bias

        # Linear layer 2
        self.linear2_weight = albert_layer.ffn_output.weight
        self.linear2_bias = albert_layer.ffn_output.bias

        # Layer norm 1
        self.norm1_eps = albert_layer.attention.LayerNorm.eps
        self.norm1_weight = albert_layer.attention.LayerNorm.weight
        self.norm1_bias = albert_layer.attention.LayerNorm.bias

        # Layer norm 2
        self.norm2_weight = albert_layer.full_layer_layer_norm.weight
        self.norm2_bias = albert_layer.full_layer_layer_norm.bias

        # Model hyper parameters
        self.num_heads = albert_layer.attention.num_attention_heads
        self.embed_dim = albert_layer.attention.all_head_size

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

    def forward(self, hidden_states, attention_mask, *_):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        if hidden_states.is_nested:
            attention_mask = None

        if attention_mask is not None:
            # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
            # 0->false->keep this token -inf->true->mask this token
            attention_mask = attention_mask.bool()
            attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
            seqlen = attention_mask.shape[1]
            lengths = torch.sum(~attention_mask, 1)
            if not all([l == seqlen for l in lengths]):
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
            attention_mask = None

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
