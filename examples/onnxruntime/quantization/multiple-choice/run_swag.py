#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Optional

import datasets
import numpy as np
import transformers
from datasets import load_dataset
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import check_min_version

from optimum.onnxruntime import ORTModelForMultipleChoice, ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, QuantizationConfig
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor
from optimum.onnxruntime.preprocessors.passes import (
    ExcludeGeLUNodes,
    ExcludeLayerNormNodes,
    ExcludeNodeAfter,
    ExcludeNodeFollowedBy,
)
from optimum.onnxruntime.utils import evaluation_loop


# Will error if the minimal version of Transformers is not installed. The version of transformers must be >= 4.19.0
# as the export to onnx of multiple choice topologies was added in this release. Remove at your own risks.
check_min_version("4.19.0")

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
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
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

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


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

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Downloading and loading the swag dataset from the hub.
        raw_datasets = load_dataset(
            "swag",
            "regular",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"

    # Preprocessing the datasets.
    def preprocess_function(examples, tokenizer: PreTrainedTokenizerBase):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=min(data_args.max_seq_length, tokenizer.model_max_length),
            padding="max_length",
        )
        # Un-flatten
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name or model_args.model_name_or_path)

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

    # Export the model
    model = ORTModelForMultipleChoice.from_pretrained(model_args.model_name_or_path, export=True)

    # Create the quantizer
    quantizer = ORTQuantizer.from_pretrained(model)

    ranges = None
    quantization_preprocessor = None
    if apply_static_quantization:
        # Preprocess the calibration dataset
        if "train" not in raw_datasets:
            raise ValueError("Static quantization requires a train dataset for calibration")
        calibration_dataset = raw_datasets["train"]
        if optim_args.num_calibration_samples is not None:
            num_calibration_samples = min(len(calibration_dataset), optim_args.num_calibration_samples)
            calibration_dataset = calibration_dataset.select(range(num_calibration_samples))
        with training_args.main_process_first(desc="Running tokenizer on the calibration dataset"):
            calibration_dataset = calibration_dataset.map(
                partial(preprocess_function, tokenizer=tokenizer),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

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

        # Create a quantization preprocessor to determine the nodes to exclude when applying static quantization
        quantization_preprocessor = QuantizationPreprocessor()
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
    model = ORTModelForMultipleChoice.from_pretrained(quantized_model_path, provider=optim_args.execution_provider)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Preprocess the evaluation dataset
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="Running tokenizer on the validation dataset"):
            eval_dataset = eval_dataset.map(
                partial(preprocess_function, tokenizer=tokenizer),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

        outputs = evaluation_loop(
            model=model,
            dataset=eval_dataset,
            label_names=["label"],
            compute_metrics=compute_metrics,
        )

        # Save evaluation metrics
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(outputs.metrics, f, indent=4, sort_keys=True)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
