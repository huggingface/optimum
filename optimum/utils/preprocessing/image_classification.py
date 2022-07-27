from functools import partial
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import FeatureExtractionMixin, ImageClassificationPipeline

from evaluate import combine, evaluator

from .base import DatasetProcessing


class ImageClassificationProcessing(DatasetProcessing):
    def __init__(self, **kwargs):
        if "secondary" in kwargs["data_keys"]:
            raise ValueError("Only one data column is supported for image-classification.")
        else:
            kwargs["data_keys"]["secondary"] = None

        super().__init__(**kwargs)

        if not isinstance(self.preprocessor, FeatureExtractionMixin):
            raise ValueError(
                f"Preprocessor is expected to be a feature extractor, provided {type(self.preprocessor)}."
            )

    def load_datasets(self):
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(path=self.dataset_path, name=self.dataset_name)

        max_eval_samples = 100  # TODO remove this

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
        if max_eval_samples is not None:
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(max_eval_samples))
        eval_dataset = eval_dataset.align_labels_with_mapping(self.config.label2id, self.ref_keys[0])

        datasets_dict = {"eval": eval_dataset}

        if self.static_quantization:
            assert self.calibration_split

            calibration_dataset = raw_datasets[self.calibration_split]
            if self.num_calibration_samples is not None:
                calibration_dataset = calibration_dataset.select(range(self.num_calibration_samples))

            # Run the preprocessor on the calibration dataset. Note that this will load images.
            calibration_dataset = calibration_dataset.map(
                partial(preprocess_function),
                batched=True,
                load_from_cache_file=True,
                desc="Running feature extractor on calibration dataset",
            )

            columns_to_remove = raw_datasets.column_names[self.calibration_split]
            columns_to_remove = [name for name in columns_to_remove if name not in self.preprocessor.model_input_names]
            calibration_dataset = calibration_dataset.remove_columns(columns_to_remove)

            datasets_dict["calibration"] = calibration_dataset

        return datasets_dict

    def run_evaluation(self, eval_dataset: Dataset, pipeline: ImageClassificationPipeline, metrics: List[str]):
        combined_metrics = combine(metrics)

        task_evaluator = evaluator("image-classification")

        results = task_evaluator.compute(
            model_or_pipeline=pipeline,
            data=eval_dataset,
            metric=combined_metrics,
            input_column=self.data_keys["primary"],
            label_column=self.ref_keys[0],
        )

        results.pop("latency", None)
        results.pop("throughput", None)

        return results

    def get_pipeline_kwargs(self):
        return {}
