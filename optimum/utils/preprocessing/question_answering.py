# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Question answering processing."""

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from .base import TaskProcessor


if TYPE_CHECKING:
    pass


class QuestionAnsweringProcessing(TaskProcessor):
    ACCEPTED_PREPROCESSOR_CLASSES = (PreTrainedTokenizerBase,)
    DEFAULT_DATASET_ARGS = "squad_v2"
    DEFAUL_DATASET_DATA_KEYS = {"question": "question", "context": "context"}
    ALLOWED_DATA_KEY_NAMES = {"question", "context"}
    DEFAULT_REF_KEYS = ["answers"]

    def create_defaults_and_kwargs_from_preprocessor_kwargs(
        self, preprocessor_kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if preprocessor_kwargs is None:
            preprocessor_kwargs = {}
        kwargs = copy.deepcopy(preprocessor_kwargs)
        defaults = {}

        pad_on_right = self.preprocessor.padding_side == "right"
        max_seq_length = min(self.preprocessor.model_max_length, 384)
        max_seq_length = kwargs.pop("max_length", max_seq_length)
        stride = min(max_seq_length // 2, 128)

        defaults["padding"] = kwargs.pop("padding", "max_length")
        defaults["truncation"] = kwargs.pop("truncation", "only_second" if pad_on_right else "only_first")
        defaults["max_length"] = max_seq_length
        defaults["stride"] = kwargs.pop("stride", stride)
        defaults["return_overflowing_tokens"] = kwargs.pop("return_overflowing_tokens", False)
        defaults["return_offsets_mapping"] = kwargs.pop("return_offsets_mapping", False)
        return defaults, kwargs

    def dataset_processing_func(
        self, example: Dict[str, Any], data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        example[data_keys["question"]] = example[data_keys["question"]].lstrip()
        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = self.preprocessor.padding_side == "right"

        tokenized_inputs = self.preprocessor(
            text=example[data_keys["question"] if pad_on_right else data_keys["context"]],
            text_pair=example[data_keys["context"] if pad_on_right else data_keys["question"]],
            **self.defaults,
            **self.preprocessor_kwargs,
        )
        return tokenized_inputs

    def try_to_guess_data_keys(self, column_names: List[str]) -> Optional[Dict[str, str]]:
        question_key_name = None
        context_key_name = None

        for name in column_names:
            if question_key_name is None and "question" in name:
                question_key_name = name
            if context_key_name is None and ("sentence" in name or "context" in name or "answer" in name):
                context_key_name = name

        if question_key_name is not None and context_key_name is not None:
            return {"question": question_key_name, "context": context_key_name}
        return None

    def try_to_guess_ref_keys(self, column_names: List[str]) -> Optional[List[str]]:
        for name in column_names:
            if "answer" in name:
                return [name]
