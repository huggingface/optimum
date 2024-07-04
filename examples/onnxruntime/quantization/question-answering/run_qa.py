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
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from transformers import AutoTokenizer, EvalPrediction, HfArgumentParser, PreTrainedTokenizer, TrainingArguments
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, QuantizationConfig
from optimum.onnxruntime.modeling_ort import ORTModelForQuestionAnswering
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor
from optimum.onnxruntime.preprocessors.passes import (
    ExcludeGeLUNodes,
    ExcludeLayerNormNodes,
    ExcludeNodeAfter,
    ExcludeNodeFollowedBy,
)
from optimum.onnxruntime.utils import evaluation_loop


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0")

require_version(
    "datasets>=1.8.0", "To fix: pip install -r examples/onnxruntime/quantization/question-answering/requirements.txt"
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

    quantization_approach: str = field(
        default="dynamic",
        metadata={"help": "The quantization approach. Supported approach are static and dynamic."},
    )
    per_channel: bool = field(
        default=False,
        metadata={"help": "Whether to quantize the weights per channel."},
    )
    reduce_range: bool = field(
        default=False,
        metadata={
            "help": "Whether to quantize the weights with 7-bits. It may improve the accuracy for some models running "
            "on non-VNNI machine, especially for per-channel mode."
        },
    )
    calibration_method: str = field(
        default="minmax",
        metadata={
            "help": "The method chosen to calculate the activation quantization parameters using the calibration "
            "dataset. Current supported calibration methods are minmax, entropy and percentile."
        },
    )
    num_calibration_samples: int = field(
        default=100,
        metadata={"help": "Number of examples to use for the calibration step resulting from static quantization."},
    )
    num_calibration_shards: int = field(
        default=1,
        metadata={
            "help": "How many shards to split the calibration dataset into. Useful for the entropy and percentile "
            "calibration method."
        },
    )
    calibration_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size for the calibration step."},
    )
    calibration_histogram_percentile: float = field(
        default=99.999,
        metadata={"help": "The percentile used for the percentile calibration method."},
    )
    calibration_moving_average: bool = field(
        default=False,
        metadata={
            "help": "Whether to compute the moving average of the minimum and maximum values for the minmax "
            "calibration method."
        },
    )
    calibration_moving_average_constant: float = field(
        default=0.01,
        metadata={
            "help": "Constant smoothing factor to use when computing the moving average of the minimum and maximum "
            "values. Effective only when the selected calibration method is minmax and `calibration_moving_average` is "
            "set to True."
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

    # TODO: currently onnxruntime put external data in different path than the model proto, which will cause problem on re-loading it.
    # https://github.com/microsoft/onnxruntime/issues/12576
    use_external_data_format: bool = field(
        default=False,
        metadata={"help": "Whether to use external data format to store model whose size is >= 2Gb."},
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

    logger.info(f"Optimization with the following parameters {optim_args}")

    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
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
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

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

    # Training preprocessing
    def prepare_train_features(examples, tokenizer: PreTrainedTokenizer):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
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
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # Validation preprocessing
    def prepare_validation_features(examples, tokenizer: PreTrainedTokenizer):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
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
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
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

    apply_static_quantization = optim_args.quantization_approach == "static"

    # Create the quantization configuration containing all the quantization parameters
    qconfig = QuantizationConfig(
        is_static=apply_static_quantization,
        format=QuantFormat.QDQ if apply_static_quantization else QuantFormat.QOperator,
        mode=QuantizationMode.QLinearOps if apply_static_quantization else QuantizationMode.IntegerOps,
        activations_dtype=QuantType.QInt8 if apply_static_quantization else QuantType.QUInt8,
        weights_dtype=QuantType.QInt8,
        per_channel=optim_args.per_channel,
        reduce_range=optim_args.reduce_range,
        operators_to_quantize=["MatMul", "Add"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Export the model
    model = ORTModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, export=True)

    # Create the quantizer
    quantizer = ORTQuantizer.from_pretrained(model)

    # Create the calibration dataset used for the static quantization calibration step
    if apply_static_quantization:
        if "train" not in raw_datasets:
            raise ValueError("Static quantization requires a train dataset for calibration")
        calibration_dataset = raw_datasets["train"]
        if optim_args.num_calibration_samples is not None:
            calibration_dataset = calibration_dataset.select(range(optim_args.num_calibration_samples))
        calibration_dataset = calibration_dataset.map(
            partial(prepare_train_features, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on the calibration dataset",
        )
        # Number of samples might increase during Feature Creation, we select the specified number of samples
        if optim_args.num_calibration_samples is not None:
            calibration_dataset = calibration_dataset.select(range(optim_args.num_calibration_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        eval_dataset = eval_examples.map(
            partial(prepare_validation_features, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on the validation dataset",
        )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        predict_dataset = predict_examples.map(
            partial(prepare_validation_features, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on the prediction dataset",
        )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    ranges = None
    # Create a quantization preprocessor to determine the nodes to exclude
    quantization_preprocessor = QuantizationPreprocessor()
    if apply_static_quantization:
        # Remove the unnecessary columns of the calibration dataset before the calibration step
        calibration_dataset = quantizer.clean_calibration_dataset(calibration_dataset)

        # Create the calibration configuration given the selected calibration method
        if optim_args.calibration_method == "entropy":
            calibration_config = AutoCalibrationConfig.entropy(calibration_dataset)
        elif optim_args.calibration_method == "percentile":
            calibration_config = AutoCalibrationConfig.percentiles(
                calibration_dataset,
                percentile=optim_args.calibration_histogram_percentile,
            )
        else:
            calibration_config = AutoCalibrationConfig.minmax(
                calibration_dataset,
                optim_args.calibration_moving_average,
                optim_args.calibration_moving_average_constant,
            )

        if not 1 <= optim_args.num_calibration_shards <= len(calibration_dataset):
            raise ValueError(
                f"Invalid value of number of shards {optim_args.num_calibration_shards} chosen to split the calibration"
                f" dataset, should be higher than 0 and lower or equal to the number of samples "
                f"{len(calibration_dataset)}."
            )

        for i in range(optim_args.num_calibration_shards):
            shard = calibration_dataset.shard(optim_args.num_calibration_shards, i)
            quantizer.partial_fit(
                dataset=shard,
                calibration_config=calibration_config,
                operators_to_quantize=qconfig.operators_to_quantize,
                batch_size=optim_args.calibration_batch_size,
                use_external_data_format=onnx_export_args.use_external_data_format,
            )
        ranges = quantizer.compute_ranges()

        # Exclude the nodes constituting LayerNorm
        quantization_preprocessor.register_pass(ExcludeLayerNormNodes())
        # Exclude the nodes constituting GELU
        quantization_preprocessor.register_pass(ExcludeGeLUNodes())
        # Exclude the residual connection Add nodes
        quantization_preprocessor.register_pass(ExcludeNodeAfter("Add", "Add"))
        # Exclude the Add nodes following the Gather operator
        quantization_preprocessor.register_pass(ExcludeNodeAfter("Gather", "Add"))
        # Exclude the Add nodes followed by the Softmax operator
        quantization_preprocessor.register_pass(ExcludeNodeFollowedBy("Add", "Softmax"))

    # Apply quantization on the model
    quantized_model_path = quantizer.quantize(
        save_dir=training_args.output_dir,
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
        preprocessor=quantization_preprocessor,
        use_external_data_format=onnx_export_args.use_external_data_format,
    )
    model = ORTModelForQuestionAnswering.from_pretrained(quantized_model_path, provider=optim_args.execution_provider)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        outputs = evaluation_loop(
            model=model,
            dataset=eval_dataset,
            compute_metrics=compute_metrics,
            label_names=["start_positions", "end_positions"],
        )
        predictions = post_processing_function(eval_examples, eval_dataset, outputs.predictions)
        metrics = compute_metrics(predictions)

        # Save metrics
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        outputs = evaluation_loop(
            model=model,
            dataset=predict_dataset,
            label_names=["start_positions", "end_positions"],
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
