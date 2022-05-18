from functools import partial
from typing import Dict, List

from datasets import Dataset, Metric, load_dataset
from transformers import PreTrainedTokenizerBase, TextClassificationPipeline

from .base import DatasetProcessing


class TextClassificationProcessing(DatasetProcessing):
    def __init__(self, **kwargs):
        if "secondary" not in kwargs["data_keys"]:
            kwargs["data_keys"]["secondary"] = None

        super().__init__(**kwargs)
        self.config = kwargs["config"]

    def load_datasets(self):
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(path=self.dataset_path, name=self.dataset_name)

        max_eval_samples = 100  # TODO remove this

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
        all_labels = []
        all_preds = []
        for _, inputs in enumerate(eval_dataset):
            has_labels = all(inputs.get(k) is not None for k in self.ref_keys)
            if has_labels:
                labels = tuple(inputs.get(name) for name in self.ref_keys)
                if len(labels) == 1:
                    labels = labels[0]
                else:
                    raise ValueError("Only one label supported.")
            else:
                raise ValueError("Missing labels")

            # the dataset label ids may be different from the label2id of predictions
            label_text = self.config.id2label[labels]
            label_id = self.config.label2id[label_text]
            all_labels.append(label_id)

            # we manually unroll the pipeline since it is broken
            # see https://github.com/huggingface/transformers/issues/17305
            if self.data_keys["secondary"]:
                inps = [inputs[self.data_keys["primary"]], inputs[self.data_keys["secondary"]]]
            else:
                inps = inputs[self.data_keys["primary"]]
            tokenized_inputs = pipeline.preprocess([inps])
            model_outputs = pipeline.forward(tokenized_inputs)
            preds = pipeline.postprocess(model_outputs)  # preds is a dict

            print("ref:", label_id)
            print("pred:", self.config.label2id[preds["label"]])
            print("-----")
            all_preds.append(self.config.label2id[preds["label"]])

        return all_labels, all_preds

    def get_metrics(self, predictions: List, references: List, metric: Metric):
        return metric.compute(predictions=predictions, references=references)

    def get_pipeline_kwargs(self):
        return {}
