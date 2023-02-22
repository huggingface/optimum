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
"""Image classification processing."""

from typing import TYPE_CHECKING, List, Dict, Any

import torch
from datasets import load_dataset
from evaluate import combine, evaluator
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import FeatureExtractionMixin

from .. import logging
from .base import DatasetProcessing

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset
    from evaluate import Metric
    from transformers import ImageClassificationPipeline


class ImageClassificationProcessing(DatasetProcessing):
    def __init__(self, *args, **kwargs):
        if "secondary" in kwargs["data_keys"]:
            raise ValueError("Only one data column is supported for image-classification.")
        else:
            kwargs["data_keys"]["secondary"] = None

        super().__init__(*args, **kwargs)

        if not isinstance(self.preprocessor, FeatureExtractionMixin):
            raise ValueError(
                f"Preprocessor is expected to be a feature extractor, provided {type(self.preprocessor)}."
            )

    def load_datasets(self) -> Dict[str, "Dataset"]:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(path=self.dataset_path, name=self.dataset_name)

        normalize = Normalize(mean=self.preprocessor.image_mean, std=self.preprocessor.image_std)
        transforms = Compose(
            [
                Resize(self.preprocessor.size),
                CenterCrop(self.preprocessor.size),
                ToTensor(),
                normalize,
            ]
        )

        # Preprocessing the raw_datasets
        def preprocess_function(examples):
            """Apply transforms across a batch."""
            examples["pixel_values"] = [
                transforms(image.convert("RGB")).to(torch.float32).numpy() for image in examples["image"]
            ]
            return examples

        eval_dataset = raw_datasets[self.eval_split]
        if self.max_eval_samples is not None:
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(self.max_eval_samples))

        try:
            eval_dataset = eval_dataset.align_labels_with_mapping(self.config.label2id, self.ref_keys[0])
        except Exception:
            logger.warning(
                f"\nModel label mapping: {self.config.label2id}"
                f"\nDataset label features: {eval_dataset.features[self.ref_keys[0]]}"
                f"\nCould not guarantee the model label mapping and the dataset labels match."
                f" Evaluation results may suffer from a wrong matching."
            )

        datasets_dict = {"eval": eval_dataset}

        if self.static_quantization:
            calibration_dataset = raw_datasets[self.calibration_split]
            if self.num_calibration_samples is not None:
                calibration_dataset = calibration_dataset.select(range(self.num_calibration_samples))

            # Run the preprocessor on the calibration dataset. Note that this will load images.
            calibration_dataset = calibration_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                desc="Running feature extractor on calibration dataset",
            )

            columns_to_remove = raw_datasets.column_names[self.calibration_split]
            columns_to_remove = [name for name in columns_to_remove if name not in self.preprocessor.model_input_names]
            calibration_dataset = calibration_dataset.remove_columns(columns_to_remove)

            datasets_dict["calibration"] = calibration_dataset

        return datasets_dict

    def run_evaluation(self, eval_dataset: "Dataset", pipeline: "ImageClassificationPipeline", metrics: List[str]):
        all_metrics = combine(metrics)

        task_evaluator = evaluator("image-classification")

        results = task_evaluator.compute(
            model_or_pipeline=pipeline,
            data=eval_dataset,
            metric=all_metrics,
            input_column=self.data_keys["primary"],
            label_column=self.ref_keys[0],
            label_mapping=self.config.label2id,
        )

        return results

    def get_metrics(self, predictions: List, references: List, metric: "Metric") -> Dict[str, float]:
        metrics_res = metric.compute(predictions=predictions, references=references)
        # `metric.compute` may return a dict or a number
        if not isinstance(metrics_res, dict):
            metrics_res = {metric.name: metrics_res}
        return metrics_res

    def get_pipeline_kwargs(self) -> Dict[str, Any]:
        return {}
