from functools import partial
from typing import Dict, List, Optional

from datasets import ClassLabel, Dataset, load_dataset
from transformers import PreTrainedTokenizerBase, TokenClassificationPipeline

from evaluate import combine, evaluator

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
        eval_dataset = eval_dataset.align_labels_with_mapping(
            label2id=self.config.label2id, label_column=self.ref_keys[0]
        )

        datasets_dict = {"eval": eval_dataset}

        features = eval_dataset.features

        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        # An example is labels as ["PER", "PER", , "O", "LOC", "O", "LOC", "O"].
        # `label_to_id` will map string labels to ids, here {'LOC': 0, 'O': 1, 'ORG': 2, 'PER': 3}.
        # Normally `label_to_id` would just be e.g. {0: 0, 1: 1, 2: 2, 3: 3}
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
        # Otherwise, we have to get the list of labels manually.
        labels_are_int = isinstance(features[self.ref_keys[0]].feature, ClassLabel)
        if labels_are_int:
            self.label_list = features[self.ref_keys[0]].feature.names
        else:
            self.label_list = get_label_list(raw_datasets["train"][self.ref_keys[0]])

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
        combined_metrics = combine(metrics)

        task_evaluator = evaluator("token-classification")

        results = task_evaluator.compute(
            model_or_pipeline=pipeline,
            data=eval_dataset,
            metric=combined_metrics,
            input_column=self.data_keys["primary"],
            label_column=self.ref_keys[0],
            join_by=" ",
        )

        results.pop("latency", None)
        results.pop("throughput", None)

        return results

    def get_pipeline_kwargs(self):
        res = {
            "ignore_labels": [],  # do not ignore "O"
        }
        return res
