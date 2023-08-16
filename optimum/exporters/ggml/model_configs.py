# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""
Common TensorFlow Lite configuration classes that handle most of the features for building model specific
configurations.
"""

import re
from typing import Dict, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from ...utils import DummyTextInputGenerator, logging
from .base import GgmlConfigWithPast


logger = logging.get_logger(__name__)


class TextDecoderGGMLConfig(GgmlConfigWithPast):
    """
    Handles encoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)
    MANDATORY_AXES = ("batch_size", "sequence_length", ("multiple-choice", "num_choices"))


# Original code: https://github.com/NouamaneTazi/bloomz.cpp/blob/main/convert-hf-to-ggml.py
class BloomGgmlConfig(TextDecoderGGMLConfig):
    STRUCT_HPARAM_KEYS = [
        "n_positions",
        "hidden_size",
        "multiple_of",
        "n_head",
        "n_layer",
    ]
    USE_BYTE_DECODER = False

    def get_cpp_name(self, name: str) -> str:
        conv_map = {
            "word_embeddings": "tok_embeddings",
            "word_embeddings_layernorm": "norm",
            "input_layernorm": "attention_norm",
            "self_attention.query_key_value": "attention.query_key_value",
            "self_attention.dense": "attention.wo",
            "post_attention_layernorm": "ffn_norm",
            "mlp.dense_h_to_4h": "feed_forward.w1",
            "mlp.dense_4h_to_h": "feed_forward.w2",
            "ln_f": "output_norm",
            "lm_head": "output",
        }
        if name != "lm_head.weight":
            nn = name.split(".")[1:]
        else:
            nn = name.split(".")
        if nn[0] == "h":
            nn[0] = "layers"
            mapped = conv_map[".".join(nn[2:-1])]
            name = ".".join(nn[:2] + [mapped] + nn[-1:])
        else:
            mapped = conv_map[".".join(nn[:-1])]
            name = ".".join([mapped] + nn[-1:])
        return name

    def reshape_weights(self, name: str, weights: Union[ndarray, Tensor], hparams: Dict) -> ndarray:
        if "query_key_value" in name:
            q, k, v = weights.reshape(hparams["n_head"], 3, -1).unbind(1)
            return torch.cat([q, k, v], dim=0).reshape_as(weights)
        return weights.squeeze().numpy()

    @staticmethod
    def convert_dtype(name: str, data: Union[ndarray, Tensor], ftype: int, n_dims: int) -> tuple[ndarray, int]:
        # default type is fp32
        if isinstance(data, Tensor):
            data = data.numpy()
        ftype_cur = 0
        data = data.astype(np.float32)
        if ftype == 1 and n_dims > 1:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        return data, ftype_cur


class GPTBigCodeGgmlConfig(TextDecoderGGMLConfig):
    STRUCT_HPARAM_KEYS = [
        "n_positions",
        "n_embd",
        "n_inner",
        "n_head",
        "n_layer",
    ]

    def get_cpp_name(self, name: str) -> str:
        if name == "transformer.ln_f.weight":
            name = "model/ln_f/g"
        elif name == "transformer.ln_f.bias":
            name = "model/ln_f/b"
        elif name == "transformer.wte.weight":
            name = "model/wte"
        elif name == "transformer.wpe.weight":
            name = "model/wpe"
        elif name == "lm_head.weight":
            name = "model/lm_head"
        elif re.match(r"transformer.h\.\d+\.ln_1\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/g"
        elif re.match(r"transformer.h\.\d+\.ln_1\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/b"
        elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/w"
        elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/b"
        elif re.match(r"transformer.h\.\d+\.attn\.c_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/w"
        elif re.match(r"transformer.h.\d+.attn.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/b"
        elif re.match(r"transformer.h.\d+.ln_2.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/g"
        elif re.match(r"transformer.h.\d+.ln_2.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/b"
        elif re.match(r"transformer.h.\d+.mlp.c_fc.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/w"
        elif re.match(r"transformer.h.\d+.mlp.c_fc.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/b"
        elif re.match(r"transformer.h.\d+.mlp.c_proj.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/w"
        elif re.match(r"transformer.h.\d+.mlp.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/b"
        else:
            print("Unrecognized variable name. %s", name)
        return name

    "model/h.*/attn/c_attn/w"
    "model/h.*/attn/c_proj/w"
    "model/h.*/mlp/c_fc/w"
    "model/h.*/mlp/c_proj/w"

    def reshape_weights(self, name: str, weights: Union[ndarray, Tensor], hparams: Dict) -> ndarray:
        weights = weights.squeeze().numpy()
        name_suffixes = {
            "/attn/c_attn/w",
            "/attn/c_attn/weight",
            "/attn/c_attn/b",
            "/attn/c_attn/bias",
        }

        if any(name.endswith(suffix) for suffix in name_suffixes):
            print("  Duplicate K,V heads to use MHA instead of MQA")

            embed_dim = hparams["n_embd"]
            head_dim = embed_dim // hparams["n_head"]

            # ((n_heads + 2) * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
            q, k, v = np.split(weights, (hparams["n_head"] * head_dim, (hparams["n_head"] + 1) * head_dim), axis=0)
            # duplicate k, v along the first axis (head_dim, hidden_dim) -> (n_heads * head_dim, hidden_dim)
            if len(k.shape) == 2:
                k = np.tile(k, (hparams["n_head"], 1))
                v = np.tile(v, (hparams["n_head"], 1))
            elif len(k.shape) == 1:
                k = np.tile(k, (hparams["n_head"]))
                v = np.tile(v, (hparams["n_head"]))
            # concat q, k, v along the first axis (n_heads * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
            weights = np.concatenate((q, k, v), axis=0)
        return weights

    def should_skip(self, name: str) -> bool:
        return name.endswith("attn.masked_bias") or name.endswith(".attn.bias")

    @staticmethod
    def convert_dtype(name: str, data: Union[ndarray, Tensor], ftype: int, n_dims: int) -> tuple[ndarray, int]:
        return data.astype(np.float32), ftype  # TODO fix the fp16 option

        if ftype == 0:
            name_suffixes = {
                "/g",
                "/g",
                "/w",
                "/weight",
            }

            if (
                name == "model/wte"
                or name == "model/lm_head"
                or any(name.endswith(suffix) for suffix in name_suffixes)
            ) and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype = 0
        return data, ftype
