from functools import partial
from typing import Dict, List

from datasets import Dataset, load_dataset
from evaluate import combine, evaluator
from transformers import PreTrainedTokenizerBase, TextClassificationPipeline

from .base import DatasetProcessing


class TextClassificationProcessing(DatasetProcessing):
    def __init__(self, **kwargs):
        if "secondary" not in kwargs["data_keys"]:
            kwargs["data_keys"]["secondary"] = None

        super().__init__(**kwargs)
        self.label_to_id = None

        if not isinstance(self.preprocessor, PreTrainedTokenizerBase):
            raise ValueError(f"Preprocessor is expected to be a tokenizer, provided {type(self.preprocessor)}.")

    def load_datasets(self):
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
            assert self.calibration_split
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

    def run_evaluation(self, eval_dataset: Dataset, pipeline: TextClassificationPipeline, metrics: List[str]):
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

    def get_pipeline_kwargs(self):
        return {}
