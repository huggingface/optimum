# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import functools
from typing import Tuple

import torch


def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)


# Modified from transformers.models.bloom.modeling_bloom._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    device: torch.device,
    past_key_values_length: int,
    dtype: torch.dtype = torch.bool,
) -> torch.BoolTensor:
    """
    Make causal mask used for bi-directional self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.zeros((target_length, target_length + past_key_values_length), dtype=dtype, device=device)
    seq_ids = torch.arange(target_length, device=device)

    mask[:, past_key_values_length:] = (
        (seq_ids[:, None] < seq_ids[None, :]) * torch.finfo(dtype).min
        if torch.is_floating_point(mask)
        else seq_ids[:, None] < seq_ids[None, :]
    )

    return mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)


# Modified from transformers.models.bloom.modeling_bloom._prepare_attn_mask
def _prepare_attn_mask(
    attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
) -> torch.BoolTensor:
    from transformers.models.bloom.modeling_bloom import _expand_mask

    # create causal mask
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    combined_attention_mask = None
    device = attention_mask.device
    _, src_length = input_shape

    combined_attention_mask = _make_causal_mask(
        input_shape, device=device, past_key_values_length=past_key_values_length
    )
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
    combined_attention_mask = (
        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
    )

    return combined_attention_mask


# Modified from transformers.models.llama.modeling_llama._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    from transformers.models.llama.modeling_llama import _expand_mask

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    combined_attention_mask = _make_causal_mask(
        input_shape,
        device=inputs_embeds.device,
        past_key_values_length=past_key_values_length,
        dtype=inputs_embeds.dtype,
    )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask
