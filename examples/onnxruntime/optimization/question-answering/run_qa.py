#!/usr/bin/env python
# coding=utf-8
#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import datasets
import transformers
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, EvalPrediction, HfArgumentParser, PreTrainedTokenizer, TrainingArguments
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions

from optimum.onnxruntime import ORTModelForQuestionAnswering, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime.utils import evaluation_loop


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0")

require_version(
    "datasets>=1.8.0", "To fix: pip install -r examples/onnxruntime/optimization/question-answering/requirements.txt"
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
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
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

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
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
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


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
    model = ORTModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, export=True)

    # Create the optimizer
    optimizer = ORTOptimizer.from_pretrained(model)

    # Optimize the model
    optimized_model_path = optimizer.optimize(
        optimization_config=optimization_config,
        save_dir=training_args.output_dir,
        use_external_data_format=onnx_export_args.use_external_data_format,
        one_external_file=onnx_export_args.one_external_file,
    )

    model = ORTModelForQuestionAnswering.from_pretrained(optimized_model_path, provider=optim_args.execution_provider)

    # Prepare the dataset downloading, preprocessing and metric creation to perform the evaluation and / or the
    # prediction step(s)
    if training_args.do_eval or training_args.do_predict:
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at
        # https://huggingface.co/datasets/ (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
                extension = data_args.train_file.split(".")[-1]
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
                extension = data_args.validation_file.split(".")[-1]
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
                extension = data_args.test_file.split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=model_args.cache_dir)
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc)
        #  at https://huggingface.co/docs/datasets/loading_datasets.html.

        # Preprocessing the datasets.
        # Preprocessing is slightly different for training and evaluation.
        if training_args.do_eval:
            column_names = raw_datasets["validation"].column_names
        elif training_args.do_predict:
            column_names = raw_datasets["test"].column_names
        else:
            column_names = raw_datasets["train"].column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # Validation preprocessing
        def prepare_validation_features(examples, tokenizer: PreTrainedTokenizer):
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

            # Padding side determines if we do (question|context) or (context|question).
            pad_on_right = tokenizer.padding_side == "right"

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This
            # results in one example possible giving several features when a context is long, each of those features
            # having a context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=min(data_args.max_seq_length, tokenizer.model_max_length),
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question)
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples

        # Post-processing:
        def post_processing_function(examples, features, predictions, stage="eval"):
            # Post-processing: we match the start logits and end logits to answers in the original context.
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                version_2_with_negative=data_args.version_2_with_negative,
                n_best_size=data_args.n_best_size,
                max_answer_length=data_args.max_answer_length,
                null_score_diff_threshold=data_args.null_score_diff_threshold,
                output_dir=training_args.output_dir,
                log_level=log_level,
                prefix=stage,
            )
            # Format the result to the format the metric expects.
            if data_args.version_2_with_negative:
                formatted_predictions = [
                    {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
                ]
            else:
                formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        metric = load("squad_v2" if data_args.version_2_with_negative else "squad")

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Preprocess the evaluation dataset
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        eval_dataset = eval_examples.map(
            partial(prepare_validation_features, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        outputs = evaluation_loop(
            model=model,
            dataset=eval_dataset,
            label_names=["start_positions", "end_positions"],
            compute_metrics=compute_metrics,
        )
        predictions = post_processing_function(eval_examples, eval_dataset, outputs.predictions)
        metrics = compute_metrics(predictions)

        # Save metrics
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Preprocess the test dataset
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        predict_dataset = predict_examples.map(
            partial(prepare_validation_features, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        outputs = evaluation_loop(
            model=model,
            dataset=eval_dataset,
            label_names=["start_positions", "end_positions"],
            compute_metrics=compute_metrics,
        )
        predictions = post_processing_function(predict_examples, predict_dataset, outputs.predictions)
        metrics = compute_metrics(predictions)

        # Save metrics
        with open(os.path.join(training_args.output_dir, "predict_results.json"), "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
