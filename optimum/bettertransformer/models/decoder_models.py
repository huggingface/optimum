# Copyright 2023 The HuggingFace and Meta Team.  All rights reserved.
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
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from .base import BetterTransformerBaseLayer


if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PretrainedConfig


def raise_on_head_mask(head_mask: Optional[torch.Tensor]):
    if head_mask is not None:
        raise ValueError(
            "layer_head_mask different than None is unsupported for now with BetterTransformer, please"
            "open a PR or an issue at https://github.com/huggingface/optimum."
        )


class GPT2AttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = self.wrapped_scaled_dot_product

        # gpt-2
        if config.model_type == "gpt2":
            target_dtype = self.gpt_layer.c_proj.weight.dtype
        # gpt-neo-x
        elif config.model_type == "gpt_neox":
            target_dtype = self.gpt_layer.dense.weight.dtype
        # gpt-j
        else:
            target_dtype = self.gpt_layer.out_proj.weight.dtype

        self.downcast_qk = config.model_type in ["gptj", "gpt_neox"]

        mask_value = torch.finfo(target_dtype).min
        self._mask_value = torch.full([], mask_value, dtype=target_dtype)

    def wrapped_scaled_dot_product(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ):
        raise_on_head_mask(head_mask)
        batch_size = query.shape[0]

        if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
            raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

        if batch_size == 1:
            if query.shape[2] > 1:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        else:
            query_length, key_length = query.size(-2), key.size(-2)

            # causal_mask is always [True, ..., True] otherwise, so executing this
            # is unnecessary
            if query_length > 1:
                causal_mask = self.gpt_layer.bias[:, :, key_length - query_length : key_length, :key_length].to(
                    torch.bool
                )

                causal_mask = torch.where(causal_mask, 0, self._mask_value)

                # torch.Tensor.expand does no memory copy
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
                attention_mask = causal_mask + attention_mask

            # in gpt-neo-x and gpt-j the query and keys are always in fp32
            # thus we need to cast them to the value dtype
            if self.downcast_qk:
                query = query.to(value.dtype)
                key = key.to(value.dtype)

            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        return sdpa_result, None

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return self.gpt_layer(*args, **kwargs)


class GPTNeoAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = self.wrapped_scaled_dot_product

        target_dtype = self.gpt_layer.k_proj.weight.dtype
        mask_value = torch.finfo(target_dtype).min
        self._mask_value = torch.full([], mask_value, dtype=target_dtype)

        if self.gpt_layer.bias[0][0][-1][0] == 1:
            self.attention_type = "global"
        else:
            self.attention_type = "local"

        self.scale = torch.sqrt(torch.tensor(self.gpt_layer.head_dim, dtype=torch.float32)).to(
            torch.get_default_dtype()
        )

    def wrapped_scaled_dot_product(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ):
        raise_on_head_mask(head_mask)
        query = query * self.scale
        batch_size = query.shape[0]

        if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
            raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

        if batch_size == 1 and self.attention_type == "global":
            if query.shape[2] > 1:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        else:
            query_length, key_length = query.size(-2), key.size(-2)

            causal_mask = self.gpt_layer.bias[:, :, key_length - query_length : key_length, :key_length]

            causal_mask = torch.where(causal_mask, 0, self._mask_value)
            if batch_size > 1:
                # torch.Tensor.expand does no memory copy
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

            attention_mask = causal_mask + attention_mask

            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        return sdpa_result, None

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return self.gpt_layer(*args, **kwargs)


class CodegenAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = self.wrapped_scaled_dot_product

        target_dtype = self.gpt_layer.qkv_proj.weight.dtype
        mask_value = torch.finfo(target_dtype).min
        self._mask_value = torch.full([], mask_value, dtype=target_dtype)

    def wrapped_scaled_dot_product(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ):
        raise_on_head_mask(head_mask)
        batch_size = query.shape[0]
        if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
            raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

        if batch_size == 1:
            if query.shape[2] > 1:
                # first step of the decoding
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                # in this case, which is the later decoding steps, the `causal_mask`` in
                # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/gpt2/modeling_gpt2.py#L195
                # is [True, ..., True] so actually not causal
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        else:
            query_length, key_length = query.size(-2), key.size(-2)

            # causal_mask is always [True, ..., True] otherwise, so executing this
            # is unnecessary
            if query_length > 1:
                causal_mask = self.gpt_layer.causal_mask[:, :, key_length - query_length : key_length, :key_length].to(
                    torch.bool
                )

                causal_mask = torch.where(causal_mask, 0, self._mask_value)

                # torch.Tensor.expand does no memory copy
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

                # sum masks
                attention_mask = causal_mask + attention_mask

            # in codegen the query and key are always in fp32 regardless of the dtype of the model
            # https://github.com/huggingface/transformers/blob/5b28b7833297adf65c5160a685425ddb1eee5ce2/src/transformers/models/codegen/modeling_codegen.py#L226
            query = query.to(value.dtype)
            key = key.to(value.dtype)

            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        return sdpa_result, None

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return self.gpt_layer(*args, **kwargs)


class OPTAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, opt_layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config, opt_layer)

        self.opt_layer = opt_layer

        mask_value = torch.finfo(torch.float32).min
        self._mask_value = torch.full([], mask_value, dtype=torch.float32)

        self.scale = torch.sqrt(torch.tensor(self.opt_layer.head_dim, dtype=torch.float32)).to(
            torch.get_default_dtype()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        super().forward_checker()
        raise_on_head_mask(layer_head_mask)

        if output_attentions is True:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        # TODO: raise on batch_size = 1 + padding

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        batch_size, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.opt_layer.q_proj(hidden_states) * self.opt_layer.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.opt_layer._shape(self.opt_layer.k_proj(key_value_states), -1, batch_size)
            value_states = self.opt_layer._shape(self.opt_layer.v_proj(key_value_states), -1, batch_size)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self.opt_layer._shape(self.opt_layer.k_proj(hidden_states), -1, batch_size)
            value_states = self.opt_layer._shape(self.opt_layer.v_proj(hidden_states), -1, batch_size)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self.opt_layer._shape(self.opt_layer.k_proj(hidden_states), -1, batch_size)
            value_states = self.opt_layer._shape(self.opt_layer.v_proj(hidden_states), -1, batch_size)

        if self.opt_layer.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_states = self.opt_layer._shape(query_states, tgt_len, batch_size)

        query_states = query_states * self.scale
        if batch_size == 1:
            if query_states.shape[2] > 1:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        if attn_output.size() != (batch_size, self.opt_layer.num_heads, tgt_len, self.opt_layer.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.opt_layer.num_heads, tgt_len, self.opt_layer.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # attn_output = attn_output.view(bsz, self.opt_layer.num_heads, tgt_len, self.opt_layer.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(batch_size, tgt_len, self.opt_layer.embed_dim)

        attn_output = self.opt_layer.out_proj(attn_output)

        return attn_output, None, past_key_value


class T5AttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config, layer)

        self.layer = layer

        mask_value = torch.finfo(torch.float32).min
        self._mask_value = torch.full([], mask_value, dtype=torch.float32)

        head_dim = self.layer.d_model // self.layer.n_heads  # hidden size / num attention heads
        self.scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        super().forward_checker()
        raise_on_head_mask(layer_head_mask)
        if len(self.layer.pruned_heads) > 0:
            raise ValueError(
                f"Setting `pruned_heads` is unsupported with BetterTransformer, found {self.layer.pruned_heads}."
            )
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.layer.n_heads, self.layer.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.layer.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.layer.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.layer.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.layer.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if (position_bias is None and not self.layer.has_relative_attention_bias) or (
            position_bias is not None and position_bias[0, 0, 0, 0] == 0
        ):
            if position_bias is None and not self.layer.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.layer.n_heads, real_seq_length, key_length),
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                if self.layer.gradient_checkpointing and self.layer.training:
                    position_bias.requires_grad = True

            query_states = self.scale * query_states
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
            )

        else:
            # compute scores
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            if position_bias is None:
                position_bias = self.layer.compute_bias(real_seq_length, key_length, device=scores.device)

                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

                if mask is not None:
                    position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

            scores += position_bias
            attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=self.layer.dropout, training=self.layer.training
            )  # (batch_size, n_heads, seq_length, key_length)

            # Mask heads if we want to
            if layer_head_mask is not None:
                attn_weights = attn_weights * layer_head_mask

            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
        attn_output = self.layer.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.layer.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
