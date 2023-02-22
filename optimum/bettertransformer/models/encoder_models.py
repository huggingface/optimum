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

from .base import BetterTransformerBaseLayer


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class AlbertLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, albert_layer, config):
        r"""
        A simple conversion of the ALBERT layer to its `BetterTransformer` implementation.

        Args:
            albert_layer (`torch.nn.Module`):
                The original ALBERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config, albert_layer)
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

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()

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
        return (hidden_states,)


class BertLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, bert_layer, config):
        r"""
        A simple conversion of the BERT layer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config, bert_layer)
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

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()

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
        return (hidden_states,)


class BartEncoderLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, bart_layer, config):
        r"""
        A simple conversion of the `BartEncoderLayer` to its `BetterTransformer` implementation.

        Args:
            bart_layer (`torch.nn.Module`):
                The original `BartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config, bart_layer)
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

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, position_bias=None, *_, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()

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
        return (hidden_states,)


class MBartEncoderLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, mbart_layer, config):
        r"""
        A simple conversion of the `MBartEncoderLayer` to its `BetterTransformer` implementation.
        Args:
            mbart_layer (`torch.nn.Module`):
                The original `MBartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config, mbart_layer)
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

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, position_bias=None, *_, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()

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
        return (hidden_states,)


class DistilBertLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, bert_layer, config):
        r"""
        A simple conversion of the Distill-BERTLayer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original Distill-BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config, bert_layer)
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


class WhisperEncoderLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, whisper_layer, config):
        r"""
        A simple conversion of the WhisperEncoderLayer to its `BetterTransformer` implementation.

        Args:
            whisper_layer (`torch.nn.Module`):
                The original `WhisperEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config, whisper_layer)
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
        self.keys_to_ignore.append("self_attn.k_proj.bias")

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
        attention_mask = None  # attention mask seems to be always None: https://github.com/huggingface/transformers/blob/94b3f544a1f5e04b78d87a2ae32a7ac252e22e31/src/transformers/models/whisper/modeling_whisper.py#L690

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


class ViTLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, vit_layer, config):
        r"""
        A simple conversion of the ViTLayer to its `BetterTransformer` implementation.

        Args:
            vit_layer (`torch.nn.Module`):
                The original `ViTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config, vit_layer)
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

    def forward(self, hidden_states, *_, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
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


class ViltLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, vilt_layer, config):
        r"""
        A simple conversion of the VilTLayer to its `BetterTransformer` implementation.

        Args:
            vilt_layer (`torch.nn.Module`):
                The original `VilTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config, vilt_layer)
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

    def forward(self, hidden_states, *_, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
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


class Wav2Vec2EncoderLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, wav2vec2_layer, config):
        r"""
        A simple conversion of the Wav2Vec2EncoderLayer to its `BetterTransformer` implementation.

        Args:
            wav2vec2_layer (`torch.nn.Module`):
                The original `Wav2Vec2EncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config, wav2vec2_layer)
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

    def forward(self, hidden_states, attention_mask, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()
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
        return (hidden_states,)


class FSMTEncoderLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, fsmt_layer, config):
        r"""
        A simple conversion of the FSMT Encoder layer to its `BetterTransformer` implementation.

        Args:
            fsmt_layer (`torch.nn.Module`):
                The original FSMT Layer where the weights needs to be retrieved.
        """
        super().__init__(config, fsmt_layer)
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

    def forward(self, hidden_states, attention_mask, position_bias=None, *_, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()

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
        return (hidden_states, attention_mask)


class CLIPLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, layer, config):
        r"""
        A simple conversion of the CLIPEncoderLayer to its `BetterTransformer` implementation.

        **The implementation is valid only for the vision model, that does not use `causal_attention_mask`.**

        Args:
            layer (`torch.nn.Module`):
                The original `CLIPEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config, layer)
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

    def forward(self, hidden_states, attention_mask, *_, **__):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        super().forward_checker()

        # we expect attention_mask to be None in the vision model
        if attention_mask is not None:
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

        return (hidden_states,)

    def _get_activation_function(self, config: "PretrainedConfig"):
        if hasattr(config, "vision_config") and hasattr(config, "text_config"):
            assert config.vision_config.hidden_act == config.text_config.hidden_act
            return config.vision_config.hidden_act
        else:
            return config.hidden_act
