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
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from .base import BetterTransformerBaseLayer


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class AlbertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, albert_layer, config):
        r"""
        A simple conversion of the ALBERT layer to its `BetterTransformer` implementation.

        Args:
            albert_layer (`torch.nn.Module`):
                The original ALBERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
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
        self.norm2_eps = albert_layer.full_layer_layer_norm.eps
        self.norm2_weight = albert_layer.full_layer_layer_norm.weight
        self.norm2_bias = albert_layer.full_layer_layer_norm.bias

        # Model hyper parameters
        self.num_heads = albert_layer.attention.num_attention_heads
        self.embed_dim = albert_layer.attention.all_head_size

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.original_layers_mapping = {
            "in_proj_weight": ["attention.query.weight", "attention.key.weight", "attention.value.weight"],
            "in_proj_bias": ["attention.query.bias", "attention.key.bias", "attention.value.bias"],
            "out_proj_weight": "attention.dense.weight",
            "out_proj_bias": "attention.dense.bias",
            "linear1_weight": "ffn.weight",
            "linear1_bias": "ffn.bias",
            "linear2_weight": "ffn_output.weight",
            "linear2_bias": "ffn_output.bias",
            "norm1_eps": "attention.LayerNorm.eps",
            "norm1_weight": "attention.LayerNorm.weight",
            "norm1_bias": "attention.LayerNorm.bias",
            "norm2_eps": "full_layer_layer_norm.eps",
            "norm2_weight": "full_layer_layer_norm.weight",
            "norm2_bias": "full_layer_layer_norm.bias",
        }
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.act_fn_callable = ACT2FN[self.act_fn]

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if hidden_states.is_nested:
                attention_mask = None

            if attention_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
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
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)

            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            # NOTE: In PyTorch 2.0, passing an attention_mask will automatically dispatch
            # to the "math" path and will NOT use flash attention / memory-efficient attention.
            # We should support xformers / Hazy-flash / rocm-flash directly and stop relying on PyTorch to do the work.
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=False,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
            )

            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)

            # BertSelfOutput
            attention_out = F.layer_norm(
                F.dropout(
                    F.linear(attention_out, self.out_proj_weight, self.out_proj_bias),
                    p=self.hidden_dropout_prob,
                    training=self.training,
                )
                + hidden_states,
                normalized_shape=self.norm1_weight.shape,
                weight=self.norm1_weight,
                bias=self.norm1_bias,
            )

            # BertIntermediate
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))

            # BertOutput
            hidden_states = F.layer_norm(
                attention_out
                + F.dropout(
                    F.linear(hidden_states, self.linear2_weight, self.linear2_bias),
                    p=self.hidden_dropout_prob,
                    training=self.training,
                ),
                normalized_shape=self.norm2_weight.shape,
                weight=self.norm2_weight,
                bias=self.norm2_bias,
            )

        return (hidden_states,)


class BertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, bert_layer, config):
        r"""
        A simple conversion of the BERT layer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.self.query.weight,
                    bert_layer.attention.self.key.weight,
                    bert_layer.attention.self.value.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.self.query.bias,
                    bert_layer.attention.self.key.bias,
                    bert_layer.attention.self.value.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = bert_layer.attention.output.dense.weight
        self.out_proj_bias = bert_layer.attention.output.dense.bias

        # Linear layer 1
        self.linear1_weight = bert_layer.intermediate.dense.weight
        self.linear1_bias = bert_layer.intermediate.dense.bias

        # Linear layer 2
        self.linear2_weight = bert_layer.output.dense.weight
        self.linear2_bias = bert_layer.output.dense.bias

        # Layer norm 1
        self.norm1_eps = bert_layer.attention.output.LayerNorm.eps
        self.norm1_weight = bert_layer.attention.output.LayerNorm.weight
        self.norm1_bias = bert_layer.attention.output.LayerNorm.bias

        # Layer norm 2
        self.norm2_eps = bert_layer.output.LayerNorm.eps
        self.norm2_weight = bert_layer.output.LayerNorm.weight
        self.norm2_bias = bert_layer.output.LayerNorm.bias

        # Model hyper parameters
        self.num_heads = bert_layer.attention.self.num_attention_heads
        self.embed_dim = bert_layer.attention.self.all_head_size

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.original_layers_mapping = {
            "in_proj_weight": [
                "attention.self.query.weight",
                "attention.self.key.weight",
                "attention.self.value.weight",
            ],
            "in_proj_bias": ["attention.self.query.bias", "attention.self.key.bias", "attention.self.value.bias"],
            "out_proj_weight": "attention.output.dense.weight",
            "out_proj_bias": "attention.output.dense.bias",
            "linear1_weight": "intermediate.dense.weight",
            "linear1_bias": "intermediate.dense.bias",
            "linear2_weight": "output.dense.weight",
            "linear2_bias": "output.dense.bias",
            "norm1_eps": "attention.output.LayerNorm.eps",
            "norm1_weight": "attention.output.LayerNorm.weight",
            "norm1_bias": "attention.output.LayerNorm.bias",
            "norm2_eps": "output.LayerNorm.eps",
            "norm2_weight": "output.LayerNorm.weight",
            "norm2_bias": "output.LayerNorm.bias",
        }
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.act_fn_callable = ACT2FN[self.act_fn]

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        # No check on output_attentions here as roformer relies on BertLayerBetterTransformer but does not pass output_attentions as keyword argument.
        if not self.training and not torch._C._is_any_autocast_enabled():
            if hidden_states.is_nested:
                attention_mask = None

            if attention_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
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
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)

            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            # NOTE: In PyTorch 2.0, passing an attention_mask will automatically dispatch
            # to the "math" path and will NOT use flash attention / memory-efficient attention.
            # We should support xformers / Hazy-flash / rocm-flash directly and stop relying on PyTorch to do the work.
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=False,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
            )

            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)

            # BertSelfOutput
            attention_out = F.layer_norm(
                F.dropout(
                    F.linear(attention_out, self.out_proj_weight, self.out_proj_bias),
                    p=self.hidden_dropout_prob,
                    training=self.training,
                )
                + hidden_states,
                normalized_shape=self.norm1_weight.shape,
                weight=self.norm1_weight,
                bias=self.norm1_bias,
            )

            # BertIntermediate
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))

            # BertOutput
            hidden_states = F.layer_norm(
                attention_out
                + F.dropout(
                    F.linear(hidden_states, self.linear2_weight, self.linear2_bias),
                    p=self.hidden_dropout_prob,
                    training=self.training,
                ),
                normalized_shape=self.norm2_weight.shape,
                weight=self.norm2_weight,
                bias=self.norm2_bias,
            )

        return (hidden_states,)


class BartEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, bart_layer, config):
        r"""
        A simple conversion of the `BartEncoderLayer` to its `BetterTransformer` implementation.

        Args:
            bart_layer (`torch.nn.Module`):
                The original `BartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    bart_layer.self_attn.q_proj.weight,
                    bart_layer.self_attn.k_proj.weight,
                    bart_layer.self_attn.v_proj.weight,
                ]
            )
        )

        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    bart_layer.self_attn.q_proj.bias,
                    bart_layer.self_attn.k_proj.bias,
                    bart_layer.self_attn.v_proj.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = bart_layer.self_attn.out_proj.weight
        self.out_proj_bias = bart_layer.self_attn.out_proj.bias

        # Linear layer 1
        self.linear1_weight = bart_layer.fc1.weight
        self.linear1_bias = bart_layer.fc1.bias

        # Linear layer 2
        self.linear2_weight = bart_layer.fc2.weight
        self.linear2_bias = bart_layer.fc2.bias

        # Layer norm 1
        self.norm1_eps = bart_layer.self_attn_layer_norm.eps
        self.norm1_weight = bart_layer.self_attn_layer_norm.weight
        self.norm1_bias = bart_layer.self_attn_layer_norm.bias

        # Layer norm 2
        self.norm2_eps = bart_layer.final_layer_norm.eps
        self.norm2_weight = bart_layer.final_layer_norm.weight
        self.norm2_bias = bart_layer.final_layer_norm.bias

        # Model hyper parameters
        self.num_heads = bart_layer.self_attn.num_heads
        self.embed_dim = bart_layer.self_attn.embed_dim

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.original_layers_mapping = {
            "in_proj_weight": ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
            "in_proj_bias": ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"],
            "out_proj_weight": "self_attn.out_proj.weight",
            "out_proj_bias": "self_attn.out_proj.bias",
            "linear1_weight": "fc1.weight",
            "linear1_bias": "fc1.bias",
            "linear2_weight": "fc2.weight",
            "linear2_bias": "fc2.bias",
            "norm1_eps": "self_attn_layer_norm.eps",
            "norm1_weight": "self_attn_layer_norm.weight",
            "norm1_bias": "self_attn_layer_norm.bias",
            "norm2_eps": "final_layer_norm.eps",
            "norm2_weight": "final_layer_norm.weight",
            "norm2_bias": "final_layer_norm.bias",
        }
        self.dropout = config.attention_dropout
        self.activation_dropout = config.activation_dropout
        self.attention_head_size = config.d_model // config.encoder_attention_heads
        self.act_fn_callable = ACT2FN[self.act_fn]

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: bool, position_bias=None, *_, **__):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if not hasattr(hidden_states, "original_shape"):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape

            if hidden_states.is_nested:
                attention_mask = None

            if attention_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
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

            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)

            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            # NOTE: In PyTorch 2.0, passing an attention_mask will automatically dispatch
            # to the "math" path and will NOT use flash attention / memory-efficient attention.
            # We should support xformers / Hazy-flash / rocm-flash directly and stop relying on PyTorch to do the work.
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )

            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)

            # BertSelfOutput
            attention_out = F.layer_norm(
                F.dropout(
                    F.linear(attention_out, self.out_proj_weight, self.out_proj_bias),
                    p=self.dropout,
                    training=self.training,
                )
                + hidden_states,
                normalized_shape=self.norm1_weight.shape,
                weight=self.norm1_weight,
                bias=self.norm1_bias,
            )

            # One additional dropout compared to bert
            hidden_states = F.dropout(
                self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias)),
                p=self.activation_dropout,
                training=self.training,
            )

            hidden_states = F.layer_norm(
                attention_out
                + F.dropout(
                    F.linear(hidden_states, self.linear2_weight, self.linear2_bias),
                    p=self.dropout,
                    training=self.training,
                ),
                normalized_shape=self.norm2_weight.shape,
                weight=self.norm2_weight,
                bias=self.norm2_bias,
            )
        return (hidden_states,)


class MBartEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, mbart_layer, config):
        r"""
        A simple conversion of the `MBartEncoderLayer` to its `BetterTransformer` implementation.
        Args:
            mbart_layer (`torch.nn.Module`):
                The original `MBartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    mbart_layer.self_attn.q_proj.weight,
                    mbart_layer.self_attn.k_proj.weight,
                    mbart_layer.self_attn.v_proj.weight,
                ]
            )
        )

        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    mbart_layer.self_attn.q_proj.bias,
                    mbart_layer.self_attn.k_proj.bias,
                    mbart_layer.self_attn.v_proj.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = mbart_layer.self_attn.out_proj.weight
        self.out_proj_bias = mbart_layer.self_attn.out_proj.bias

        # Linear layer 1
        self.linear1_weight = mbart_layer.fc1.weight
        self.linear1_bias = mbart_layer.fc1.bias

        # Linear layer 2
        self.linear2_weight = mbart_layer.fc2.weight
        self.linear2_bias = mbart_layer.fc2.bias

        # Layer norm 1
        self.norm1_eps = mbart_layer.self_attn_layer_norm.eps
        self.norm1_weight = mbart_layer.self_attn_layer_norm.weight
        self.norm1_bias = mbart_layer.self_attn_layer_norm.bias

        # Layer norm 2
        self.norm2_eps = mbart_layer.final_layer_norm.eps
        self.norm2_weight = mbart_layer.final_layer_norm.weight
        self.norm2_bias = mbart_layer.final_layer_norm.bias

        # Model hyper parameters
        self.num_heads = mbart_layer.self_attn.num_heads
        self.embed_dim = mbart_layer.self_attn.embed_dim

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False
        self.norm_first = True

        self.original_layers_mapping = {
            "in_proj_weight": [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ],
            "in_proj_bias": ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"],
            "out_proj_weight": "self_attn.out_proj.weight",
            "out_proj_bias": "self_attn.out_proj.bias",
            "linear1_weight": "fc1.weight",
            "linear1_bias": "fc1.bias",
            "linear2_weight": "fc2.weight",
            "linear2_bias": "fc2.bias",
            "norm1_weight": "self_attn_layer_norm.weight",
            "norm1_bias": "self_attn_layer_norm.bias",
            "norm1_eps": "self_attn_layer_norm.eps",
            "norm2_weight": "final_layer_norm.weight",
            "norm2_bias": "final_layer_norm.bias",
            "norm2_eps": "final_layer_norm.eps",
        }
        self.dropout = config.attention_dropout
        self.activation_dropout = config.activation_dropout
        self.attention_head_size = config.d_model // config.encoder_attention_heads
        self.act_fn_callable = ACT2FN[self.act_fn]

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: bool, position_bias=None, *_, **__):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if not hasattr(hidden_states, "original_shape"):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape

            if hidden_states.is_nested:
                attention_mask = None

            if attention_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
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

            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            residual = hidden_states
            hidden_states = F.layer_norm(
                hidden_states,
                normalized_shape=self.norm1_weight.shape,
                weight=self.norm1_weight,
                bias=self.norm1_bias,
            )

            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            # NOTE: In PyTorch 2.0, passing an attention_mask will automatically dispatch
            # to the "math" path and will NOT use flash attention / memory-efficient attention.
            # We should support xformers / Hazy-flash / rocm-flash directly and stop relying on PyTorch to do the work.
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )

            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)

            hidden_states = residual + F.dropout(
                F.linear(attention_out, self.out_proj_weight, self.out_proj_bias),
                p=self.dropout,
                training=self.training,
            )
            residual = hidden_states
            hidden_states = F.layer_norm(
                hidden_states,
                normalized_shape=self.norm2_weight.shape,
                weight=self.norm2_weight,
                bias=self.norm2_bias,
            )

            # One additional dropout compared to bert
            hidden_states = F.dropout(
                self.act_fn_callable(F.linear(hidden_states, self.linear1_weight, self.linear1_bias)),
                p=self.activation_dropout,
                training=self.training,
            )

            hidden_states = residual + F.dropout(
                F.linear(hidden_states, self.linear2_weight, self.linear2_bias),
                p=self.dropout,
                training=self.training,
            )

        return (hidden_states,)


class DistilBertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, bert_layer, config):
        r"""
        A simple conversion of the Distill-BERTLayer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original Distill-BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
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

        self.original_layers_mapping = {
            "in_proj_weight": ["attention.q_lin.weight", "attention.k_lin.weight", "attention.v_lin.weight"],
            "in_proj_bias": ["attention.q_lin.bias", "attention.k_lin.bias", "attention.v_lin.bias"],
            "out_proj_weight": "attention.out_lin.weight",
            "out_proj_bias": "attention.out_lin.bias",
            "linear1_weight": "ffn.lin1.weight",
            "linear1_bias": "ffn.lin1.bias",
            "linear2_weight": "ffn.lin2.weight",
            "linear2_bias": "ffn.lin2.bias",
            "norm1_weight": "sa_layer_norm.weight",
            "norm1_bias": "sa_layer_norm.bias",
            "norm2_weight": "output_layer_norm.weight",
            "norm2_bias": "output_layer_norm.bias",
        }
        self.attention_dropout = config.attention_dropout
        self.dropout = config.dropout
        self.attention_head_size = config.dim // config.n_heads
        self.act_fn_callable = ACT2FN[self.act_fn]

        self.validate_bettertransformer()

    def forward(self, hidden_states, attn_mask, output_attentions: bool, head_mask=None, *_):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if hidden_states.is_nested:
                attn_mask = None

            if attn_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                attn_mask = attn_mask.bool()
                attn_mask = torch.reshape(attn_mask, (attn_mask.shape[0], attn_mask.shape[-1]))
                seqlen = attn_mask.shape[1]
                lengths = torch.sum(~attn_mask, 1)
                if not all(l == seqlen for l in lengths):
                    hidden_states = torch._nested_tensor_from_mask(hidden_states, attn_mask)
                attn_mask = None

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
                attn_mask,
            )
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)

            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            # TODO: Kind of stupid to do that at each layer, should be fixed in transformers
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).to(dtype=query.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(query.dtype).min

            # NOTE: In PyTorch 2.0, passing an attention_mask will automatically dispatch
            # to the "math" path and will NOT use flash attention / memory-efficient attention.
            # We should support xformers / Hazy-flash / rocm-flash directly and stop relying on PyTorch to do the work.
            if self.training:
                attn_mask = None
            attention_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )

            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)

            # BertSelfOutput
            attention_out = F.layer_norm(
                F.dropout(
                    F.linear(attention_out, self.out_proj_weight, self.out_proj_bias),
                    p=self.dropout,
                    training=self.training,
                )
                + hidden_states,
                normalized_shape=self.norm1_weight.shape,
                weight=self.norm1_weight,
                bias=self.norm1_bias,
            )

            # BertIntermediate
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))

            # BertOutput
            hidden_states = F.layer_norm(
                attention_out
                + F.dropout(
                    F.linear(hidden_states, self.linear2_weight, self.linear2_bias),
                    p=self.dropout,
                    training=self.training,
                ),
                normalized_shape=self.norm2_weight.shape,
                weight=self.norm2_weight,
                bias=self.norm2_bias,
            )
        return (hidden_states,)


class ViTLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, vit_layer, config):
        r"""
        A simple conversion of the ViTLayer to its `BetterTransformer` implementation.

        Args:
            vit_layer (`torch.nn.Module`):
                The original `ViTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    vit_layer.attention.attention.query.weight,
                    vit_layer.attention.attention.key.weight,
                    vit_layer.attention.attention.value.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    vit_layer.attention.attention.query.bias,
                    vit_layer.attention.attention.key.bias,
                    vit_layer.attention.attention.value.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = vit_layer.attention.output.dense.weight
        self.out_proj_bias = vit_layer.attention.output.dense.bias

        # Linear layer 1
        self.linear1_weight = vit_layer.intermediate.dense.weight
        self.linear1_bias = vit_layer.intermediate.dense.bias

        # Linear layer 2
        self.linear2_weight = vit_layer.output.dense.weight
        self.linear2_bias = vit_layer.output.dense.bias

        # Layer norm 1
        self.norm1_eps = vit_layer.layernorm_before.eps
        self.norm1_weight = vit_layer.layernorm_before.weight
        self.norm1_bias = vit_layer.layernorm_before.bias

        # Layer norm 2
        self.norm2_eps = vit_layer.layernorm_after.eps
        self.norm2_weight = vit_layer.layernorm_after.weight
        self.norm2_bias = vit_layer.layernorm_after.bias

        # Model hyper parameters
        self.num_heads = vit_layer.attention.attention.num_attention_heads
        self.embed_dim = int(vit_layer.attention.attention.attention_head_size * self.num_heads)

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False
        self.norm_first = True

        self.original_layers_mapping = {
            "in_proj_weight": [
                "attention.attention.query.weight",
                "attention.attention.key.weight",
                "attention.attention.value.weight",
            ],
            "in_proj_bias": [
                "attention.attention.query.bias",
                "attention.attention.key.bias",
                "attention.attention.value.bias",
            ],
            "out_proj_weight": "attention.output.dense.weight",
            "out_proj_bias": "attention.output.dense.bias",
            "linear1_weight": "intermediate.dense.weight",
            "linear1_bias": "intermediate.dense.bias",
            "linear2_weight": "output.dense.weight",
            "linear2_bias": "output.dense.bias",
            "norm1_weight": "layernorm_before.weight",
            "norm1_bias": "layernorm_before.bias",
            "norm2_weight": "layernorm_after.weight",
            "norm2_bias": "layernorm_after.bias",
        }

        self.validate_bettertransformer()

    def forward(self, hidden_states, output_attentions: bool, *_, **__):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
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
        else:
            raise NotImplementedError(
                "Training and Autocast are not implemented for BetterTransformer + ViT. Please open an issue."
            )
        return (hidden_states,)


class ViltLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, vilt_layer, config):
        r"""
        A simple conversion of the VilTLayer to its `BetterTransformer` implementation.

        Args:
            vilt_layer (`torch.nn.Module`):
                The original `VilTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    vilt_layer.attention.attention.query.weight,
                    vilt_layer.attention.attention.key.weight,
                    vilt_layer.attention.attention.value.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    vilt_layer.attention.attention.query.bias,
                    vilt_layer.attention.attention.key.bias,
                    vilt_layer.attention.attention.value.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = vilt_layer.attention.output.dense.weight
        self.out_proj_bias = vilt_layer.attention.output.dense.bias

        # Linear layer 1
        self.linear1_weight = vilt_layer.intermediate.dense.weight
        self.linear1_bias = vilt_layer.intermediate.dense.bias

        # Linear layer 2
        self.linear2_weight = vilt_layer.output.dense.weight
        self.linear2_bias = vilt_layer.output.dense.bias

        # Layer norm 1
        self.norm1_eps = vilt_layer.layernorm_before.eps
        self.norm1_weight = vilt_layer.layernorm_before.weight
        self.norm1_bias = vilt_layer.layernorm_before.bias

        # Layer norm 2
        self.norm2_eps = vilt_layer.layernorm_after.eps
        self.norm2_weight = vilt_layer.layernorm_after.weight
        self.norm2_bias = vilt_layer.layernorm_after.bias

        # Model hyper parameters
        self.num_heads = vilt_layer.attention.attention.num_attention_heads
        self.embed_dim = int(vilt_layer.attention.attention.attention_head_size * self.num_heads)

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False
        self.norm_first = True

        self.original_layers_mapping = {
            "in_proj_weight": [
                "attention.attention.query.weight",
                "attention.attention.key.weight",
                "attention.attention.value.weight",
            ],
            "in_proj_bias": [
                "attention.attention.query.bias",
                "attention.attention.key.bias",
                "attention.attention.value.bias",
            ],
            "out_proj_weight": "attention.output.dense.weight",
            "out_proj_bias": "attention.output.dense.bias",
            "linear1_weight": "intermediate.dense.weight",
            "linear1_bias": "intermediate.dense.bias",
            "linear2_weight": "output.dense.weight",
            "linear2_bias": "output.dense.bias",
            "norm1_weight": "layernorm_before.weight",
            "norm1_bias": "layernorm_before.bias",
            "norm2_weight": "layernorm_after.weight",
            "norm2_bias": "layernorm_after.bias",
        }

        self.validate_bettertransformer()

    def forward(self, hidden_states, layer_head_mask, output_attentions: bool, *_, **__):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
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
        else:
            raise NotImplementedError(
                "Training and Autocast are not implemented for BetterTransformer + Vilt. Please open an issue."
            )
        return (hidden_states,)


class Wav2Vec2EncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, wav2vec2_layer, config):
        r"""
        A simple conversion of the Wav2Vec2EncoderLayer to its `BetterTransformer` implementation.

        Args:
            wav2vec2_layer (`torch.nn.Module`):
                The original `Wav2Vec2EncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    wav2vec2_layer.attention.q_proj.weight,
                    wav2vec2_layer.attention.k_proj.weight,
                    wav2vec2_layer.attention.v_proj.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    wav2vec2_layer.attention.q_proj.bias,
                    wav2vec2_layer.attention.k_proj.bias,
                    wav2vec2_layer.attention.v_proj.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = wav2vec2_layer.attention.out_proj.weight
        self.out_proj_bias = wav2vec2_layer.attention.out_proj.bias

        # Linear layer 1
        self.linear1_weight = wav2vec2_layer.feed_forward.intermediate_dense.weight
        self.linear1_bias = wav2vec2_layer.feed_forward.intermediate_dense.bias

        # Linear layer 2
        self.linear2_weight = wav2vec2_layer.feed_forward.output_dense.weight
        self.linear2_bias = wav2vec2_layer.feed_forward.output_dense.bias

        # Layer norm 1
        self.norm1_eps = wav2vec2_layer.layer_norm.eps
        self.norm1_weight = wav2vec2_layer.layer_norm.weight
        self.norm1_bias = wav2vec2_layer.layer_norm.bias

        # Layer norm 2
        self.norm2_eps = wav2vec2_layer.final_layer_norm.eps
        self.norm2_weight = wav2vec2_layer.final_layer_norm.weight
        self.norm2_bias = wav2vec2_layer.final_layer_norm.bias

        # Model hyper parameters
        self.num_heads = wav2vec2_layer.attention.num_heads
        self.embed_dim = wav2vec2_layer.attention.embed_dim

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.original_layers_mapping = {
            "in_proj_weight": ["attention.q_proj.weight", "attention.k_proj.weight", "attention.v_proj.weight"],
            "in_proj_bias": ["attention.q_proj.bias", "attention.k_proj.bias", "attention.v_proj.bias"],
            "out_proj_weight": "attention.out_proj.weight",
            "out_proj_bias": "attention.out_proj.bias",
            "linear1_weight": "feed_forward.intermediate_dense.weight",
            "linear1_bias": "feed_forward.intermediate_dense.bias",
            "linear2_weight": "feed_forward.output_dense.weight",
            "linear2_bias": "feed_forward.output_dense.bias",
            "norm1_weight": "layer_norm.weight",
            "norm1_bias": "layer_norm.bias",
            "norm1_eps": "layer_norm.eps",
            "norm2_weight": "final_layer_norm.weight",
            "norm2_bias": "final_layer_norm.bias",
            "norm2_eps": "final_layer_norm.eps",
        }

        if config.do_stable_layer_norm:
            self.norm_first = True

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: bool, **__):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if hidden_states.is_nested:
                attention_mask = None

            if attention_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                attention_mask = attention_mask.bool()
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
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
        else:
            raise NotImplementedError(
                "Training and Autocast are not implemented for BetterTransformer + Wav2Vec2. Please open an issue."
            )
        return (hidden_states,)


class FSMTEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, fsmt_layer, config):
        r"""
        A simple conversion of the FSMT Encoder layer to its `BetterTransformer` implementation.

        Args:
            fsmt_layer (`torch.nn.Module`):
                The original FSMT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    fsmt_layer.self_attn.q_proj.weight,
                    fsmt_layer.self_attn.k_proj.weight,
                    fsmt_layer.self_attn.v_proj.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    fsmt_layer.self_attn.q_proj.bias,
                    fsmt_layer.self_attn.k_proj.bias,
                    fsmt_layer.self_attn.v_proj.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = fsmt_layer.self_attn.out_proj.weight
        self.out_proj_bias = fsmt_layer.self_attn.out_proj.bias

        # Linear layer 1
        self.linear1_weight = fsmt_layer.fc1.weight
        self.linear1_bias = fsmt_layer.fc1.bias

        # Linear layer 2
        self.linear2_weight = fsmt_layer.fc2.weight
        self.linear2_bias = fsmt_layer.fc2.bias

        # Layer norm 1
        self.norm1_eps = fsmt_layer.self_attn_layer_norm.eps
        self.norm1_weight = fsmt_layer.self_attn_layer_norm.weight
        self.norm1_bias = fsmt_layer.self_attn_layer_norm.bias

        # Layer norm 2
        self.norm2_eps = fsmt_layer.final_layer_norm.eps
        self.norm2_weight = fsmt_layer.final_layer_norm.weight
        self.norm2_bias = fsmt_layer.final_layer_norm.bias

        # Model hyper parameters
        self.num_heads = fsmt_layer.self_attn.num_heads
        self.embed_dim = fsmt_layer.self_attn.embed_dim

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.original_layers_mapping = {
            "in_proj_weight": ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
            "in_proj_bias": ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"],
            "out_proj_weight": "self_attn.out_proj.weight",
            "out_proj_bias": "self_attn.out_proj.bias",
            "linear1_weight": "fc1.weight",
            "linear1_bias": "fc1.bias",
            "linear2_weight": "fc2.weight",
            "linear2_bias": "fc2.bias",
            "norm1_weight": "self_attn_layer_norm.weight",
            "norm1_bias": "self_attn_layer_norm.bias",
            "norm2_weight": "final_layer_norm.weight",
            "norm2_bias": "final_layer_norm.bias",
        }

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: bool, position_bias=None, *_, **__):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if not hasattr(hidden_states, "original_shape"):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape

            if hidden_states.is_nested:
                attention_mask = None

            if attention_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))

                # FSMT swaps the first two axis before calling the encoder stack
                # Reference: https://github.com/huggingface/transformers/blob/699e90437f984d69ad3c9b891dd2e9d0fc2cffe4/src/transformers/models/fsmt/modeling_fsmt.py#L508
                if hidden_states.shape[0] != attention_mask.shape[0]:
                    hidden_states = hidden_states.transpose(1, 0)
                    original_shape = hidden_states.shape

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

            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            raise NotImplementedError(
                "Training and Autocast are not implemented for BetterTransformer + FSMT. Please open an issue."
            )

        return (hidden_states, attention_mask)


class ProphetNetEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, prophetnet_layer, config):
        r"""
        A simple conversion of the ProphetNet Encoder layer to its `BetterTransformer` implementation.

        Args:
            prophet_net_layer (`torch.nn.Module`):
                The original ProphetNet Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.config = config
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    prophetnet_layer.self_attn.query_proj.weight,
                    prophetnet_layer.self_attn.key_proj.weight,
                    prophetnet_layer.self_attn.value_proj.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    prophetnet_layer.self_attn.query_proj.bias,
                    prophetnet_layer.self_attn.key_proj.bias,
                    prophetnet_layer.self_attn.value_proj.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = prophetnet_layer.self_attn.out_proj.weight
        self.out_proj_bias = prophetnet_layer.self_attn.out_proj.bias

        # Linear layer 1
        self.linear1_weight = prophetnet_layer.feed_forward.intermediate.weight
        self.linear1_bias = prophetnet_layer.feed_forward.intermediate.bias

        # Linear layer 2
        self.linear2_weight = prophetnet_layer.feed_forward.output.weight
        self.linear2_bias = prophetnet_layer.feed_forward.output.bias

        # Layer norm 1
        self.norm1_eps = prophetnet_layer.self_attn_layer_norm.eps
        self.norm1_weight = prophetnet_layer.self_attn_layer_norm.weight
        self.norm1_bias = prophetnet_layer.self_attn_layer_norm.bias

        # Layer norm 2
        self.norm2_eps = prophetnet_layer.feed_forward_layer_norm.eps
        self.norm2_weight = prophetnet_layer.feed_forward_layer_norm.weight
        self.norm2_bias = prophetnet_layer.feed_forward_layer_norm.bias

        # Model hyper parameters
        self.num_heads = prophetnet_layer.self_attn.num_attn_heads
        self.embed_dim = prophetnet_layer.self_attn.head_dim * self.num_heads

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.original_layers_mapping = {
            "in_proj_weight": [
                "self_attn.query_proj.weight",
                "self_attn.key_proj.weight",
                "self_attn.value_proj.weight",
            ],
            "in_proj_bias": ["self_attn.query_proj.bias", "self_attn.key_proj.bias", "self_attn.value_proj.bias"],
            "out_proj_weight": "self_attn.out_proj.weight",
            "out_proj_bias": "self_attn.out_proj.bias",
            "linear1_weight": "feed_forward.intermediate.weight",
            "linear1_bias": "feed_forward.intermediate.bias",
            "linear2_weight": "feed_forward.output.weight",
            "linear2_bias": "feed_forward.output.bias",
            "norm1_weight": "self_attn_layer_norm.weight",
            "norm1_bias": "self_attn_layer_norm.bias",
            "norm2_weight": "feed_forward_layer_norm.weight",
            "norm2_bias": "feed_forward_layer_norm.bias",
        }

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: bool, *_, **__):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if not hasattr(hidden_states, "original_shape"):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape

            if hidden_states.is_nested:
                attention_mask = None

            if attention_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
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
            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            raise ValueError(
                "Training and Autocast are not implemented for BetterTransformer + ProphetNet. Please open an issue."
            )

        return (hidden_states,)


class CLIPLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, layer, config):
        r"""
        A simple conversion of the CLIPEncoderLayer to its `BetterTransformer` implementation.

        **The implementation is valid only for the vision model, that does not use `causal_attention_mask`.**

        Args:
            layer (`torch.nn.Module`):
                The original `CLIPEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    layer.self_attn.q_proj.bias,
                    layer.self_attn.k_proj.bias,
                    layer.self_attn.v_proj.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = layer.self_attn.out_proj.weight
        self.out_proj_bias = layer.self_attn.out_proj.bias

        # Linear layer 1
        self.linear1_weight = layer.mlp.fc1.weight
        self.linear1_bias = layer.mlp.fc1.bias

        # Linear layer 2
        self.linear2_weight = layer.mlp.fc2.weight
        self.linear2_bias = layer.mlp.fc2.bias

        # Layer norm 1
        self.norm1_eps = layer.layer_norm1.eps
        self.norm1_weight = layer.layer_norm1.weight
        self.norm1_bias = layer.layer_norm1.bias

        # Layer norm 2
        self.norm2_eps = layer.layer_norm2.eps
        self.norm2_weight = layer.layer_norm2.weight
        self.norm2_bias = layer.layer_norm2.bias

        # Model hyper parameters
        self.num_heads = layer.self_attn.num_heads
        self.embed_dim = layer.self_attn.embed_dim

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False
        self.norm_first = True

        self.original_layers_mapping = {
            "in_proj_weight": ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
            "in_proj_bias": ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"],
            "out_proj_weight": "self_attn.out_proj.weight",
            "out_proj_bias": "self_attn.out_proj.bias",
            "linear1_weight": "mlp.fc1.weight",
            "linear1_bias": "mlp.fc1.bias",
            "linear2_weight": "mlp.fc2.weight",
            "linear2_bias": "mlp.fc2.bias",
            "norm1_eps": "layer_norm1.eps",
            "norm1_weight": "layer_norm1.weight",
            "norm1_bias": "layer_norm1.bias",
            "norm2_eps": "layer_norm2.eps",
            "norm2_weight": "layer_norm2.weight",
            "norm2_bias": "layer_norm2.bias",
        }

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, causal_attention_mask, output_attentions: bool, *_, **__):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            # we expect attention_mask to be None in the vision model
            if attention_mask is not None or causal_attention_mask is not None:
                raise ValueError(
                    "Please do not use attention masks when using `BetterTransformer` converted vision models"
                )

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
        else:
            raise NotImplementedError(
                "Training and Autocast are not implemented for BetterTransformer + CLIP. Please open an issue."
            )

        return (hidden_states,)

    def _get_activation_function(self, config: "PretrainedConfig"):
        if hasattr(config, "vision_config") and hasattr(config, "text_config"):
            assert config.vision_config.hidden_act == config.text_config.hidden_act
            return config.vision_config.hidden_act
        else:
            return config.hidden_act
