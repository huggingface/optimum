from functools import partial
from typing import Dict, List, Optional

import tqdm
from datasets import ClassLabel, Dataset, Metric, load_dataset
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

        features = raw_datasets["train"].features

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

    def run_inference(self, eval_dataset: Dataset, pipeline: TokenClassificationPipeline):
        all_labels = [[self.label_list[l] for l in label if l != -100] for label in eval_dataset[self.ref_keys[0]]]
        all_preds = []
        for i, data in enumerate(tqdm(eval_dataset, miniters=max(1, len(eval_dataset)))):
            inputs = " ".join(data[self.data_keys["primary"]])
            res = pipeline(inputs)

            # BatchEncoding.word_ids may be wrong as we joined words with " ", so let's populate it ourselves
            token_to_word_id = []
            for j, word in enumerate(data[self.data_keys["primary"]]):
                preprocessed_inputs = pipeline.preprocess(word)
                n_tokens = len([k for k in preprocessed_inputs.word_ids(0) if k != None])  # exclude None
                token_to_word_id.extend([j] * n_tokens)

            # the pipeline may give as output labeled tokens that are part of the same word, keep track
            # of the indexing to match the true labels on words
            index_tokens_word_start = []

            for j, word_index in enumerate(token_to_word_id):
                if j == 0:
                    index_tokens_word_start.append(j)
                elif word_index != token_to_word_id[j - 1]:
                    index_tokens_word_start.append(j)

            # keep only predictions that correspond to the beginning of a word
            preds = [res[index]["entity"] for index in index_tokens_word_start]

            assert len(preds) == len(all_labels[i])
            all_preds.append(preds)

        return all_labels, all_preds

    def get_metrics(self, predictions: List, references: List, metric: Metric):
        metrics_res = metric.compute(predictions=predictions, references=references)

        if metric.name == "seqeval":
            metrics_res = {
                "precision": metrics_res["overall_precision"],
                "recall": metrics_res["overall_recall"],
                "f1": metrics_res["overall_f1"],
                "accuracy": metrics_res["overall_accuracy"],
            }
        # `metric.compute` may return a dict or a number
        elif not isinstance(metrics_res, dict):
            metrics_res = {metric.name: metrics_res}

        return metrics_res

    def get_pipeline_kwargs(self):
        res = {
            "ignore_labels": [],  # do not ignore "O"
        }
        return res
