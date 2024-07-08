#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import datasets
import numpy as np
import transformers
from datasets import ClassLabel, load_dataset
from evaluate import load
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizer, TrainingArguments
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from optimum.onnxruntime import ORTModelForTokenClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime.utils import evaluation_loop


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0")

require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/onnxruntime/optimization/token-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a validation/test file.")
        else:
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    optimization_level: Optional[int] = field(
        default=1,
        metadata={
            "help": "Optimization level performed by ONNX Runtime of the loaded graph."
            "0 will disable all optimizations."
            "1 will enable basic optimizations."
            "2 will enable basic and extended optimizations, including complex node fusions applied to the nodes "
            "assigned to the CPU or CUDA execution provider, making the resulting optimized graph hardware dependent."
            "99 will enable all available optimizations including layout optimizations."
        },
    )
    optimize_with_onnxruntime_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only use ONNX Runtime to optimize the model and no graph fusion in Python."
            "Graph fusion might require offline, Python scripts, to be run."
        },
    )
    optimize_for_gpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to optimize the model for GPU inference. The optimized graph might contain operators for "
            "GPU or CPU only when optimization_level > 1."
        },
    )
    execution_provider: str = field(
        default="CPUExecutionProvider",
        metadata={"help": "ONNX Runtime execution provider to use for inference."},
    )


@dataclass
class OnnxExportArguments:
    """
    Arguments to decide how the ModelProto will be saved.
    """

    use_external_data_format: bool = field(
        default=False,
        metadata={"help": "Whether to use external data format to store model whose size is >= 2Gb."},
    )
    one_external_file: bool = field(
        default=True,
        metadata={"help": "When `use_external_data_format=True`, whether to save all tensors to one external file."},
    )


def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, OptimizationArguments, OnnxExportArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, optim_args, onnx_export_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, optim_args, onnx_export_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if (
        optim_args.optimization_level > 1
        and optim_args.optimize_for_gpu
        and optim_args.execution_provider == "CPUExecutionProvider"
    ):
        raise ValueError(
            f"Optimization level is set at {optim_args.optimization_level} and "
            f"GPU optimization will be done, although the CPU execution provider "
            f"was selected. Use --execution_provider CUDAExecutionProvider."
        )

    if (
        optim_args.optimization_level > 1
        and not optim_args.optimize_for_gpu
        and optim_args.execution_provider == "CUDAExecutionProvider"
    ):
        raise ValueError(
            f"Optimization level is set at {optim_args.optimization_level} and "
            f"CPU optimization will be done, although the GPU execution provider "
            f"was selected. Remove the argument --execution_provider CUDAExecutionProvider."
        )

    logger.info(f"Optimization with the following parameters {optim_args}")

    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    os.makedirs(training_args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name or model_args.model_name_or_path)

    # Create the optimization configuration containing all the optimization parameters
    optimization_config = OptimizationConfig(
        optimization_level=optim_args.optimization_level,
        optimize_with_onnxruntime_only=optim_args.optimize_with_onnxruntime_only,
        optimize_for_gpu=optim_args.optimize_for_gpu,
    )

    # Export the model
    model = ORTModelForTokenClassification.from_pretrained(model_args.model_name_or_path, export=True)

    # Create the optimizer
    optimizer = ORTOptimizer.from_pretrained(model)

    # Optimize the model
    optimizer.optimize(
        optimization_config=optimization_config,
        save_dir=training_args.output_dir,
        use_external_data_format=onnx_export_args.use_external_data_format,
        one_external_file=onnx_export_args.one_external_file,
    )

    # Prepare the dataset downloading, preprocessing and metric creation to perform the evaluation and / or the
    # prediction step(s)
    if training_args.do_eval or training_args.do_predict:
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at
        # https://huggingface.co/datasets/ (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
        else:
            data_files = {}
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
                extension = data_args.validation_file.split(".")[-1]
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
                extension = data_args.test_file.split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

        if training_args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            column_names = raw_datasets["validation"].column_names
        else:
            column_names = raw_datasets["train"].column_names

        if data_args.text_column_name is not None:
            text_column_name = data_args.text_column_name
        elif "tokens" in column_names:
            text_column_name = "tokens"
        else:
            text_column_name = column_names[0]

        if data_args.label_column_name is not None:
            label_column_name = data_args.label_column_name
        elif f"{data_args.task_name}_tags" in column_names:
            label_column_name = f"{data_args.task_name}_tags"
        else:
            label_column_name = column_names[1]

        if training_args.do_eval:
            # Preprocess the evaluation dataset
            eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

            try:
                eval_dataset = eval_dataset.align_labels_with_mapping(
                    label2id=model.config.label2id, label_column=label_column_name
                )
            except Exception:
                logger.warning(
                    f"\nModel label mapping: {model.config.label2id}"
                    f"\nDataset label features: {eval_dataset.features[label_column_name]}"
                    f"\nCould not guarantee the model label mapping and the dataset labels match."
                    f" Evaluation results may suffer from a wrong matching."
                )

            features = eval_dataset.features
        else:
            features = raw_datasets["train"].features

        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
        # Otherwise, we have to get the list of labels manually.
        labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
        if labels_are_int:
            label_list = features[label_column_name].feature.names
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(raw_datasets["validation"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}

        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []
        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

        # Tokenize all texts and align the labels with them.
        def tokenize_and_align_labels(examples, tokenizer: PreTrainedTokenizer):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding="max_length",
                truncation=True,
                max_length=data_args.max_seq_length,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label_to_id[label[word_idx]])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if data_args.label_all_tokens:
                            label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        # Metrics
        metric = load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            if data_args.return_entity_level_metrics:
                # Unpack nested dictionaries
                final_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            final_results[f"{key}_{n}"] = v
                    else:
                        final_results[key] = value
                return final_results
            else:
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_dataset = eval_dataset.map(
            partial(tokenize_and_align_labels, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on the validation dataset",
        )

        outputs = evaluation_loop(
            model=model,
            dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Save evaluation metrics
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(outputs.metrics, f, indent=4, sort_keys=True)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Preprocess the test dataset
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            partial(tokenize_and_align_labels, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on the prediction dataset",
        )

        outputs = evaluation_loop(
            model=model,
            dataset=predict_dataset,
            compute_metrics=compute_metrics,
        )
        predictions = np.argmax(outputs.predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, outputs.label_ids)
        ]

        # Save test metrics
        with open(os.path.join(training_args.output_dir, "predict_results.json"), "w") as f:
            json.dump(outputs.metrics, f, indent=4, sort_keys=True)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        with open(output_predictions_file, "w") as writer:
            for prediction in true_predictions:
                writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
