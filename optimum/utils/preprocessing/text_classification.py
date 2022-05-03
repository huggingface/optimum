from functools import partial
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset, Metric, load_dataset
from transformers import PretrainedConfig, PreTrainedTokenizerBase, TextClassificationPipeline
from transformers.trainer_pt_utils import nested_concat

from .base import DatasetProcessing


class TextClassificationProcessing(DatasetProcessing):
    def __init__(self, **kwargs):
        if "secondary" in kwargs["data_keys"]:
            raise ValueError("Only one data column is supported for now.")
        else:
            kwargs["data_keys"]["secondary"] = None

        super().__init__(**kwargs)
        self.config = kwargs["config"]

    def load_datasets(self):
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(path=self.dataset_path, name=self.dataset_name)

        max_eval_samples = 20  # TODO remove this

        # Preprocessing the raw_datasets
        def preprocess_function(
            examples, data_keys: Dict[str, str], tokenizer: PreTrainedTokenizerBase, max_length: int
        ):
            # Tokenize the texts

            tokenized_inputs = tokenizer(
                text=examples[data_keys["primary"]],
                text_pair=examples[data_keys["secondary"]] if data_keys["secondary"] else None,
                padding="max_length",
                max_length=min(max_length, tokenizer.model_max_length),
                truncation=True,
            )
            return tokenized_inputs

        eval_dataset = raw_datasets[self.eval_split]
        if max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        datasets_dict = {"eval": eval_dataset}

        if self.static_quantization:
            assert self.calibration_split
            # Run the tokenizer on the calibration dataset
            calibration_dataset = raw_datasets[self.calibration_split].map(
                partial(
                    preprocess_function,
                    tokenizer=self.tokenizer,
                    data_keys=self.data_keys,
                    max_length=self.max_seq_length,
                ),
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on calibration dataset",
            )

            # TODO careful we don't remove required
            columns_to_remove = raw_datasets.column_names[self.calibration_split]
            calibration_dataset = calibration_dataset.remove_columns(columns_to_remove)

            if self.num_calibration_samples is not None:
                calibration_dataset = calibration_dataset.select(range(self.num_calibration_samples))

            datasets_dict["calibration"] = calibration_dataset

        return datasets_dict

    def run_inference(self, eval_dataset: Dataset, pipeline: TextClassificationPipeline):
        all_labels = None
        all_preds = None
        for _, inputs in enumerate(eval_dataset):
            has_labels = all(inputs.get(k) is not None for k in self.ref_keys)
            if has_labels:
                labels = tuple(np.array([inputs.get(name)]) for name in self.ref_keys)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                raise ValueError("Missing labels")

            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

            # TODO support several columns (e.g. mnli)
            preds = pipeline(inputs[self.data_keys["primary"]])
            if len(preds) == 1:
                preds = preds[0]

            # Is always `label` as an output of pipeline?
            pred_id = np.array([self.config.label2id[preds["label"]]])

            all_preds = pred_id if all_preds is None else nested_concat(all_preds, pred_id, padding_index=-100)

        return all_labels, all_preds

    def get_metrics(self, predictions: List, references: List, metric: Metric):
        return metric.compute(predictions=predictions, references=references)

    def get_pipeline_kwargs(self):
        return {}
