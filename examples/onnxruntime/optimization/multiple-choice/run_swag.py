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
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import check_min_version

from optimum.onnxruntime import ORTModelForMultipleChoice, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
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
        and model_args.execution_provider == "CPUExecutionProvider"
    ):
        raise ValueError(
            f"Optimization level is set at {optim_args.optimization_level} and "
            f"GPU optimization will be done, although the CPU execution provider "
            f"was selected. Use --execution_provider CUDAExecutionProvider."
        )

    if (
        optim_args.optimization_level > 1
        and not optim_args.optimize_for_gpu
        and model_args.execution_provider == "CUDAExecutionProvider"
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
    model = ORTModelForMultipleChoice.from_pretrained(model_args.model_name_or_path, export=True)

    # Create the optimizer
    optimizer = ORTOptimizer.from_pretrained(model)

    # Optimize the model
    optimized_model_path = optimizer.optimize(
        optimization_config=optimization_config,
        save_dir=training_args.output_dir,
        use_external_data_format=onnx_export_args.use_external_data_format,
        one_external_file=onnx_export_args.one_external_file,
    )

    model = ORTModelForMultipleChoice.from_pretrained(
        optimized_model_path,
        provider=optim_args.execution_provider,
    )

    if training_args.do_eval:
        # Prepare the dataset downloading, preprocessing and metric creation to perform the evaluation and / or the
        # prediction step(s)
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

        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

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

        # Preprocess the evaluation dataset
        with training_args.main_process_first(desc="Running tokenizer on the validation dataset"):
            eval_dataset = eval_dataset.map(
                partial(preprocess_function, tokenizer=tokenizer),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

        # Metric
        def compute_metrics(eval_predictions):
            predictions, label_ids = eval_predictions
            preds = np.argmax(predictions, axis=1)
            return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

        # Evaluation
        logger.info("*** Evaluate ***")

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
