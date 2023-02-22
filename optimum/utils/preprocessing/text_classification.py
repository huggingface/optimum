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
from typing import TYPE_CHECKING, Any, Dict, List

from datasets import load_dataset
from evaluate import combine, evaluator
from transformers import PreTrainedTokenizerBase

from .base import DatasetProcessing

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import TextClassificationPipeline


class TextClassificationProcessing(DatasetProcessing):
    def __init__(self, *args, **kwargs):
        if "secondary" not in kwargs["data_keys"]:
            kwargs["data_keys"]["secondary"] = None

        super().__init__(*args, **kwargs)
        self.label_to_id = None

        if not isinstance(self.preprocessor, PreTrainedTokenizerBase):
            raise ValueError(f"Preprocessor is expected to be a tokenizer, provided {type(self.preprocessor)}.")

    def load_datasets(self) -> Dict[str, "Dataset"]:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(path=self.dataset_path, name=self.dataset_name)

        # Preprocessing the raw_datasets
        def preprocess_function(examples, data_keys: Dict[str, str], tokenizer: PreTrainedTokenizerBase):
            # Tokenize the texts

            tokenized_inputs = tokenizer(
                text=examples[data_keys["primary"]],
                text_pair=examples[data_keys["secondary"]] if data_keys["secondary"] else None,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            return tokenized_inputs

        eval_dataset = raw_datasets[self.eval_split]
        if self.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(self.max_eval_samples))
        eval_dataset = eval_dataset.align_labels_with_mapping(self.config.label2id, self.ref_keys[0])

        datasets_dict = {"eval": eval_dataset}

        if self.static_quantization:
            # Run the tokenizer on the calibration dataset
            calibration_dataset = raw_datasets[self.calibration_split].map(
                partial(
                    preprocess_function,
                    tokenizer=self.preprocessor,
                    data_keys=self.data_keys,
                ),
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on calibration dataset",
            )

            columns_to_remove = raw_datasets.column_names[self.calibration_split]
            columns_to_remove = [name for name in columns_to_remove if name not in self.preprocessor.model_input_names]
            calibration_dataset = calibration_dataset.remove_columns(columns_to_remove)

            if self.num_calibration_samples is not None:
                calibration_dataset = calibration_dataset.select(range(self.num_calibration_samples))

            datasets_dict["calibration"] = calibration_dataset

        return datasets_dict

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
