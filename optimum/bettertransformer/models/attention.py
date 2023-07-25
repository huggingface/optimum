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

from typing import Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import _expand_mask as _llama_expand_mask
from transformers.models.llama.modeling_llama import _make_causal_mask as _llama_make_causal_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def raise_on_head_mask(head_mask: Optional[torch.Tensor]):
    if head_mask is not None:
        raise ValueError(
            "layer_head_mask different than None is unsupported for now with BetterTransformer, please"
            "open a PR or an issue at https://github.com/huggingface/optimum."
        )


# Adapted from transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn
def gpt2_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    raise_on_head_mask(head_mask)
    batch_size = query.shape[0]

    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    # in gpt-neo-x and gpt-j the query and keys are always in fp32
    # thus we need to cast them to the value dtype
    if self.downcast_qk:
        query = query.to(value.dtype)
        key = key.to(value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length, key_length = query.size(-2), key.size(-2)

        # causal_mask is always [True, ..., True] otherwise, so executing this
        # is unnecessary
        if query_length > 1:
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)

            causal_mask = torch.where(causal_mask, 0, mask_value)

            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            if attention_mask is not None:
                attention_mask = causal_mask + attention_mask

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    # in gpt-neo-x and gpt-j the query and keys are always in fp32
    # thus we need to cast them to the value dtype
    if self.downcast_qk:
        sdpa_result = sdpa_result.to(value.dtype)

    return sdpa_result, None


# Adapted from transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention._attn
def gpt_neo_wrapped_scaled_dot_product(
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

    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if (batch_size == 1 or self.training) and self.attention_type == "global":
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length, key_length = query.size(-2), key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        causal_mask = torch.where(causal_mask, 0, mask_value)
        if batch_size > 1:
            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

        if attention_mask is not None:
            attention_mask = causal_mask + attention_mask

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    return sdpa_result, None


# Adapted from transformers.models.codegen.modeling_codegen.CodeGenAttention._attn
def codegen_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    raise_on_head_mask(head_mask)
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    # in codegen the query and key are always in fp32 regardless of the dtype of the model
    # https://github.com/huggingface/transformers/blob/5b28b7833297adf65c5160a685425ddb1eee5ce2/src/transformers/models/codegen/modeling_codegen.py#L226
    query = query.to(value.dtype)
    key = key.to(value.dtype)

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            # first step of the decoding
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            # in this case, which is the later decoding steps, the `causal_mask`` in
            # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/gpt2/modeling_gpt2.py#L195
            # is [True, ..., True] so actually not causal
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length, key_length = query.size(-2), key.size(-2)

        # causal_mask is always [True, ..., True] otherwise, so executing this
        # is unnecessary
        if query_length > 1:
            causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)

            causal_mask = torch.where(causal_mask, 0, mask_value)

            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

            # we use torch.min to avoid having tensor(-inf)
            attention_mask = torch.min(causal_mask, attention_mask)

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    return sdpa_result, None


# Adapted from transformers.models.opt.modeling_opt.OPTAttention.forward
def opt_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    raise_on_head_mask(layer_head_mask)

    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    # TODO: raise on batch_size = 1 + padding

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    batch_size, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
        value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    query_states = self._shape(query_states, tgt_len, batch_size)

    query_states = query_states * self.scale

    dropout_p = self.dropout if self.training else 0.0
    if batch_size == 1 or self.training:
        if query_states.shape[2] > 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    if attn_output.size() != (batch_size, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, past_key_value


# Adapted from transformers.models.t5.modeling_t5.T5Attention.forward
def t5_forward(
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
    raise_on_head_mask(layer_head_mask)

    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")
    if len(self.pruned_heads) > 0:
        raise ValueError(f"Setting `pruned_heads` is unsupported with BetterTransformer, found {self.pruned_heads}.")
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
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

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
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    dropout_p = self.dropout if self.training else 0.0
    query_states = self.scale * query_states
    if position_bias is None and not self.has_relative_attention_bias:
        if mask is None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
        elif mask is not None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=mask, dropout_p=dropout_p, is_causal=False
            )

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length),
                device=value_states.device,
                dtype=value_states.dtype,
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=value_states.device)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.has_relative_attention_bias:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=position_bias, dropout_p=dropout_p, is_causal=False
            )
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=position_bias, dropout_p=dropout_p, is_causal=False
        )

    attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    return outputs


# Adapted from transformers.models.bart.modeling_bart.BartAttention.forward
def bart_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    raise_on_head_mask(layer_head_mask)
    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states)
    # get key, value proj
    # `past_key_value[0].shape[2] == key_value_states.shape[1]`
    # is checking that the `sequence_length` of the `past_key_value` is the same as
    # the provided `key_value_states` to support prefix tuning
    if is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    query_states = self._shape(query_states, tgt_len, bsz)
    key_states = key_states
    value_states = value_states

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
    )

    if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, past_key_value


# Adapted from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _llama_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    # We do not care about the attention mask in the batch size = 1 case
    if attention_mask.size(0) > 1:
        if input_shape[-1] > 1:
            combined_attention_mask = _llama_make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _llama_expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
    else:
        if input_shape[-1] > 1 and attention_mask is not None and attention_mask[0][0] == 0:
            raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    return combined_attention_mask


def llama_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    if bsz == 1 or self.training:
        # BEWARE: at this stage, attention_mask is not the same as in transformers llama
        if query_states.shape[2] > 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
            )
    else:
        # At this stage, **attention_mask is the same** as in transformers llama
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
