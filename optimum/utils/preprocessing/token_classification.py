from functools import partial
from typing import Dict, List

from datasets import Dataset, load_dataset
from evaluate import combine, evaluator
from transformers import PreTrainedTokenizerBase, TokenClassificationPipeline

from .base import DatasetProcessing


class TokenClassificationProcessing(DatasetProcessing):
    def __init__(self, **kwargs):
        if "secondary" in kwargs["data_keys"]:
            raise ValueError("Only one data column is supported for token-classification.")
        else:
            kwargs["data_keys"]["secondary"] = None

        super().__init__(**kwargs)

        if not isinstance(self.preprocessor, PreTrainedTokenizerBase):
            raise ValueError(f"Preprocessor is expected to be a tokenizer, provided {type(self.preprocessor)}.")

    def load_datasets(self) -> Dict:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(path=self.dataset_path, name=self.dataset_name)

        # Preprocessing the raw_datasets
        def preprocess_function(examples, data_keys: Dict[str, str], tokenizer: PreTrainedTokenizerBase):
            # Tokenize the texts
            tokenized_inputs = tokenizer(
                text=examples[data_keys["primary"]],
                text_pair=examples[data_keys["secondary"]] if data_keys["secondary"] else None,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                is_split_into_words=True,
            )
            return tokenized_inputs

        eval_dataset = raw_datasets[self.eval_split]
        if self.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(self.max_eval_samples))

        try:
            eval_dataset = eval_dataset.align_labels_with_mapping(
                label2id=self.config.label2id, label_column=self.ref_keys[0]
            )
        except Exception:
            print(
                f"\nModel label mapping: {self.config.label2id}"
                f"\nDataset label features: {eval_dataset.features[self.ref_keys[0]]}"
                f"\nCould not guarantee the model label mapping and the dataset labels match."
                f" Evaluation results may suffer from a wrong matching."
            )

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
                load_from_cache_file=False,
                desc="Running tokenizer on calibration dataset",
            )

            columns_to_remove = raw_datasets.column_names[self.calibration_split]
            columns_to_remove = [name for name in columns_to_remove if name not in self.preprocessor.model_input_names]
            calibration_dataset = calibration_dataset.remove_columns(columns_to_remove)

            if self.num_calibration_samples is not None:
                calibration_dataset = calibration_dataset.select(range(self.num_calibration_samples))

            datasets_dict["calibration"] = calibration_dataset

        return datasets_dict

    def run_evaluation(self, eval_dataset: Dataset, pipeline: TokenClassificationPipeline, metrics: List[str]):
        all_metrics = combine(metrics)

        task_evaluator = evaluator("token-classification")

        results = task_evaluator.compute(
            model_or_pipeline=pipeline,
            data=eval_dataset,
            metric=all_metrics,
            input_column=self.data_keys["primary"],
            label_column=self.ref_keys[0],
            join_by=" ",
        )

        return results

    def get_pipeline_kwargs(self):
        res = {
            "ignore_labels": [],  # do not ignore "O"
        }
        return res
