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
"""Text classification processing."""

from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from datasets import load_dataset, Dataset, DatasetDict
from evaluate import combine, evaluator
from transformers import PreTrainedTokenizerBase

from .base import TaskProcessing

if TYPE_CHECKING:
    from transformers import TextClassificationPipeline


class TextClassificationProcessing(TaskProcessing):
    ACCEPTED_PREPROCESSOR_CLASSES = (PreTrainedTokenizerBase,)

    def dataset_processing_func(self, example: Dict[str, Any], data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        tokenized_inputs = self.preprocessor(
            text=example[data_keys["primary"]],
            text_pair=example[data_keys["secondary"]] if "secondary" in data_keys else None,
            padding="max_length",
            max_length=self.preprocessor.model_max_length,
            truncation=True,
        )
        return tokenized_inputs


    def try_to_guess_data_keys(self, column_names: List[str]) -> Optional[Dict[str, str]]:
        primary_key_name = None
        for name in column_names:
            if "sentence" in name or "text" in name:
                primary_key_name = name
                break
        if primary_key_name is None and len(column_names) == 2:
            if "label" in column_names[0]:
                primary_key_name = column_names[1]
            elif "label" in column_names[1]:
                primary_key_name = column_names[0]

        return {"primary": primary_key_name} if primary_key_name is not None else None
    

    def try_to_guess_ref_keys(self, column_names: List[str]) -> Optional[List[str]]:
        for name in column_names:
            if "label" in name:
                return [name]

    def load_dataset(
        self, 
        *args, 
        data_keys: Optional[Dict[str, str]] = None, 
        ref_keys: Optional[List[str]] = None, 
        only_keep_necessary_columns: bool = False,
        **kwargs,
    ) -> Union[DatasetDict, Dataset]:
        dataset = super().load_dataset(*args, data_keys=data_keys, only_keep_necessary_columns=only_keep_necessary_columns, ref_keys=ref_keys, **kwargs)

        # TODO: do we want to do that here?
        # eval_dataset = eval_dataset.align_labels_with_mapping(self.config.label2id, self.ref_keys[0])

        return dataset

    def run_evaluation(self, eval_dataset: "Dataset", pipeline: "TextClassificationPipeline", metrics: List[str]):
        all_metrics = combine(metrics)

        task_evaluator = evaluator("text-classification")

        results = task_evaluator.compute(
            model_or_pipeline=pipeline,
            data=eval_dataset,
            metric=all_metrics,
            input_column=self.data_keys["primary"],
            label_column=self.ref_keys[0],
            label_mapping=self.config.label2id,
        )

        return results

    def get_pipeline_kwargs(self) -> Dict[str, Any]:
        return {}
