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
"""Token classification processing."""

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase

from .base import TaskProcessor


if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


class TokenClassificationProcessing(TaskProcessor):
    ACCEPTED_PREPROCESSOR_CLASSES = (PreTrainedTokenizerBase,)
    DEFAULT_DATASET_ARGS = "conll2003"
    DEFAUL_DATASET_DATA_KEYS = {"primary": "tokens"}
    ALLOWED_DATA_KEY_NAMES = {"primary"}
    DEFAULT_REF_KEYS = ["ner_tags", "pos_tags", "chunk_tags"]

    def create_defaults_and_kwargs_from_preprocessor_kwargs(
        self, preprocessor_kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if preprocessor_kwargs is None:
            preprocessor_kwargs = {}
        kwargs = copy.deepcopy(preprocessor_kwargs)
        defaults = {}
        defaults["padding"] = kwargs.pop("padding", "max_length")
        defaults["truncation"] = kwargs.pop("truncation", True)
        defaults["max_length"] = kwargs.pop("max_length", self.preprocessor.model_max_length)
        defaults["is_split_into_words"] = kwargs.pop("is_split_into_words", True)

        return defaults, kwargs

    def dataset_processing_func(
        self, example: Dict[str, Any], data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        tokenized_inputs = self.preprocessor(
            text=example[data_keys["primary"]],
            **self.defaults,
            **self.preprocessor_kwargs,
        )
        return tokenized_inputs

    def try_to_guess_data_keys(self, column_names: List[str]) -> Optional[Dict[str, str]]:
        primary_key_name = None
        for name in column_names:
            if "token" in name or "text" in name or "sentence" in name:
                primary_key_name = name
                break
        return {"primary": primary_key_name} if primary_key_name is not None else None

    def try_to_guess_ref_keys(self, column_names: List[str]) -> Optional[List[str]]:
        ref_keys = []
        for name in column_names:
            if "tag" in name:
                ref_keys.append(name)
        return ref_keys if ref_keys else None

    def load_dataset(
        self,
        path: str,
        data_keys: Optional[Dict[str, str]] = None,
        ref_keys: Optional[List[str]] = None,
        only_keep_necessary_columns: bool = False,
        load_smallest_split: bool = False,
        num_samples: Optional[int] = None,
        shuffle: bool = False,
        **load_dataset_kwargs,
    ) -> Union["DatasetDict", "Dataset"]:
        if data_keys is not None and data_keys.get("secondary", None) is not None:
            raise ValueError("Only one data column is supported for token-classification.")
        dataset = super().load_dataset(
            path,
            data_keys=data_keys,
            ref_keys=ref_keys,
            only_keep_necessary_columns=only_keep_necessary_columns,
            load_smallest_split=load_smallest_split,
            num_samples=num_samples,
            shuffle=shuffle,
            **load_dataset_kwargs,
        )
        # TODO: do we want to do that here?
        # eval_dataset = eval_dataset.align_labels_with_mapping(self.config.label2id, self.ref_keys[0])
        return dataset
