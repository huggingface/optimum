# Copyright 2022 The HuggingFace and Meta Team.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file ehidden_statescept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ehidden_statespress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn


class T5LayerFast(nn.Module):
    def __init__(self, t5_layer):
        r"""
        A simple conversion of the T5-layer to its `Fast` implementation.

        Args:
            t5_layer (`torch.nn.Module`):
                The original T5 Layer where the weights needs to be retrieved.
        """
        super().__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    t5_layer.layer[0].SelfAttention.q.weight,
                    t5_layer.layer[0].SelfAttention.k.weight,
                    t5_layer.layer[0].SelfAttention.v.weight,
                ]
            )
        )

        self.in_proj_bias = nn.Parameter(torch.zeros(3 * t5_layer.layer[0].SelfAttention.q.weight.shape[0]))

        # Out proj layer
        self.out_proj_weight = t5_layer.layer[0].SelfAttention.o.weight
        self.out_proj_bias = torch.zeros(self.out_proj_weight.shape[0])

        # Linear layer 1
        self.linear1_weight = t5_layer.layer[1].DenseReluDense.wi.weight
        self.linear1_bias = torch.zeros(self.linear1_weight.shape[0])

        # Linear layer 2
        self.linear2_weight = t5_layer.layer[1].DenseReluDense.wo.weight
        self.linear2_bias = torch.zeros(self.linear2_weight.shape[0])

        # Layer norm 1
        self.norm1_eps = t5_layer.layer[0].layer_norm.variance_epsilon
        self.norm1_weight = t5_layer.layer[0].layer_norm.weight
        self.norm1_bias = torch.zeros(self.norm1_weight.shape[0])

        # Layer norm 2
        self.norm2_weight = t5_layer.layer[1].layer_norm.weight
        self.norm2_bias = torch.zeros(self.norm2_weight.shape[0])

        # Model hyper parameters
        self.num_heads = t5_layer.layer[0].SelfAttention.n_heads
        self.embed_dim = t5_layer.layer[0].SelfAttention.d_model

        if hasattr(t5_layer.layer[0].SelfAttention, "relative_attention_bias"):
            self.position_bias = t5_layer.layer[0].SelfAttention.relative_attention_bias
        else:
            self.position_bias = None

        del t5_layer

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

    def forward(self, hidden_states, attention_mask, position_bias=None, *_, **__):
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
            True,  # TODO use_gelu. make it not hardcoded
            False,  # norm_first, currently not supported
            self.norm1_eps,
            self.norm1_weight,
            self.norm1_bias,
            self.norm2_weight,
            self.norm2_bias,
            self.linear1_weight,
            self.linear1_bias,
            self.linear2_weight,
            self.linear2_bias,
            attention_mask,  # TODO fihidden_states this
        )
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return (hidden_states, None, self.position_bias)

    # return (nested_hidden_states,)
