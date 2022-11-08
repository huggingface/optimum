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

from ..base import BetterTransformerBaseLayer, bettertransformer_forward_checker


class DistilBertLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, bert_layer, config):
        r"""
        A simple conversion of the Distill-BERTLayer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original Distill-BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.q_lin.weight,
                    bert_layer.attention.k_lin.weight,
                    bert_layer.attention.v_lin.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.q_lin.bias,
                    bert_layer.attention.k_lin.bias,
                    bert_layer.attention.v_lin.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = bert_layer.attention.out_lin.weight
        self.out_proj_bias = bert_layer.attention.out_lin.bias

        # Linear layer 1
        self.linear1_weight = bert_layer.ffn.lin1.weight
        self.linear1_bias = bert_layer.ffn.lin1.bias

        # Linear layer 2
        self.linear2_weight = bert_layer.ffn.lin2.weight
        self.linear2_bias = bert_layer.ffn.lin2.bias

        # Layer norm 1
        self.norm1_eps = bert_layer.sa_layer_norm.eps
        self.norm1_weight = bert_layer.sa_layer_norm.weight
        self.norm1_bias = bert_layer.sa_layer_norm.bias

        # Layer norm 2
        self.norm2_eps = bert_layer.output_layer_norm.eps
        self.norm2_weight = bert_layer.output_layer_norm.weight
        self.norm2_bias = bert_layer.output_layer_norm.bias

        # Model hyper parameters
        self.num_heads = bert_layer.attention.n_heads
        self.embed_dim = bert_layer.attention.dim

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.validate_bettertransformer()

    def forward(self, x, attn_mask, head_mask=None, output_attentions=None, *_):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()

        if x.is_nested:
            attn_mask = None

        if attn_mask is not None:
            # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
            # 0->false->keep this token -inf->true->mask this token
            attn_mask = attn_mask.bool()
            attn_mask = torch.reshape(attn_mask, (attn_mask.shape[0], attn_mask.shape[-1]))
            seqlen = attn_mask.shape[1]
            lengths = torch.sum(~attn_mask, 1)
            if not all([l == seqlen for l in lengths]):
                x = torch._nested_tensor_from_mask(x, attn_mask)
            attn_mask = None

        x = torch._transformer_encoder_layer_fwd(
            x,
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
            attn_mask,
        )
        if x.is_nested and self.is_last_layer:
            x = x.to_padded_tensor(0.0)
        return (x,)
