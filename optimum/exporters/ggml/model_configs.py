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
    CONV_MAP = {
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
