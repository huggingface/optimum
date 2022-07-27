from functools import partial
from typing import Dict, List

from datasets import Dataset, load_dataset
from transformers import PretrainedConfig, PreTrainedTokenizerBase, TextClassificationPipeline
from transformers.pipelines.text_classification import ClassificationFunction

from evaluate import load

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

        # Labels
        if not self.task_args["is_regression"]:
            label_list = raw_datasets[self.eval_split].features[self.ref_keys[0]].names
            num_labels = len(label_list)
        else:
            num_labels = 1

        if (
            self.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and not self.task_args["is_regression"]
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in self.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                self.label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
            else:
                print(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels:"
                    f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
                )

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

            all_labels.append(labels)

            # we manually unroll the pipeline since it is broken
            # see https://github.com/huggingface/transformers/issues/17305
            inps = {"text": inputs[self.data_keys["primary"]]}
            if self.data_keys["secondary"]:
                inps["text_pair"] = inputs[self.data_keys["secondary"]]

            kwargs = {"padding": "max_length"}
            tokenized_inputs = pipeline.preprocess(inps, **kwargs)
            model_outputs = pipeline.forward(tokenized_inputs)

            # preds is a dict. No processing function is applied as not needed for score in the regression case
            preds = pipeline.postprocess(model_outputs, function_to_apply=ClassificationFunction.NONE)

            if not self.task_args["is_regression"]:
                if self.label_to_id is not None:
                    preds = int(self.config.label2id[preds["label"]])
                    preds = self.label_to_id[preds]  # dataset label ids may be different than of the model label ids
                else:
                    preds = self.config.label2id[preds["label"]]
            else:
                preds = preds["score"]

            all_preds.append(preds)

        results = {}
        for metric_name in metrics:
            metric = load(metric_name)
            metric_res = metric.compute(predictions=all_preds, references=all_labels)
            # `metric.compute` may return a dict or a number
            if not isinstance(metric_res, dict):
                metric_res = {metric.name: metric_res}

            results.update(metric_res)

        return results

    def get_pipeline_kwargs(self):
        return {}
