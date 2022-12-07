from functools import partial
from typing import Dict, List

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase, QuestionAnsweringPipeline

from evaluate import combine, evaluator, load

from .base import DatasetProcessing


class QuestionAnsweringProcessing(DatasetProcessing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(self.preprocessor, PreTrainedTokenizerBase):
            raise ValueError(f"Preprocessor is expected to be a tokenizer, provided {type(self.preprocessor)}.")

    def load_datasets(self):
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(path=self.dataset_path, name=self.dataset_name)

        # Preprocessing the raw_datasets
        def preprocess_function(
            examples,
            data_keys: Dict[str, str],
            tokenizer: PreTrainedTokenizerBase,
        ):
            max_seq_len = min(tokenizer.model_max_length, 384)
            doc_stride = min(max_seq_len // 2, 128)

            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples[data_keys["question"]] = [q.lstrip() for q in examples[data_keys["question"]]]

            # Padding side determines if we do (question|context) or (context|question).
            pad_on_right = tokenizer.padding_side == "right"

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                text=examples[data_keys["question"] if pad_on_right else data_keys["context"]],
                text_pair=examples[data_keys["context"] if pad_on_right else data_keys["question"]],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_len,
                stride=doc_stride,
                return_overflowing_tokens=False,  # not needed as we don't care about labels
                return_offsets_mapping=False,  # not needed as we don't care about labels
                padding="max_length",
            )
            return tokenized_examples

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

    def run_evaluation(self, eval_dataset: Dataset, pipeline: QuestionAnsweringPipeline, metrics: List[str]):
        if len(metrics) == 1:
            all_metrics = load(metrics[0])
        else:
            all_metrics = combine(metrics)

        task_evaluator = evaluator("question-answering")

        results = task_evaluator.compute(
            model_or_pipeline=pipeline,
            data=eval_dataset,
            metric=all_metrics,
            question_column=self.data_keys["question"],
            context_column=self.data_keys["context"],
            label_column=self.ref_keys[0],
        )

        return results

    def get_pipeline_kwargs(self):
        return {"max_answer_len": 30, "padding": "max_length"}
