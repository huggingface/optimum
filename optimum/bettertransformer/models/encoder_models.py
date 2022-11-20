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

from .base import BetterTransformerBaseLayer


class AlbertLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, albert_layer, config):
        r"""
        A simple conversion of the ALBERT layer to its `BetterTransformer` implementation.

        Args:
            albert_layer (`torch.nn.Module`):
                The original ALBERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
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


class BertLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, bert_layer, config):
        r"""
        A simple conversion of the BERT layer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
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


class BartEncoderLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, bart_layer, config):
        r"""
        A simple conversion of the `BartEncoderLayer` to its `BetterTransformer` implementation.

        Args:
            bart_layer (`torch.nn.Module`):
                The original `BartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
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

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, position_bias=None, *_, **__):
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
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask.squeeze(1)[:, 0]
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


class WhisperEncoderLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, whisper_layer, config):
        r"""
        A simple conversion of the WhisperEncoderLayer to its `BetterTransformer` implementation.

        Args:
            whisper_layer (`torch.nn.Module`):
                The original `WhisperEncoderLayer` where the weights needs to be retrieved.
        """
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
        super().__init__(config)
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
        super().__init__(config)
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
