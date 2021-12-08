#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import transformers
from datasets import ClassLabel, load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import yaml
from optimum.intel.neural_compressor.trainer_inc import IncTrainer
from optimum.intel.neural_compressor.utils import CONFIG_NAME


os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
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
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
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
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    quantize: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply quantization."},
    )
    quantization_approach: Optional[str] = field(
        default=None,
        metadata={"help": "Quantization approach. Supported approach are static, dynamic and aware_training."},
    )
    prune: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply pruning."},
    )
    target_sparsity: Optional[float] = field(
        default=None,
        metadata={"help": "Targeted sparsity when pruning the model."},
    )
    quantization_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing the YAML configuration file used to control the quantization and "
            "tuning behavior."
        },
    )
    pruning_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing the YAML configuration file used to control the pruning behavior."
        },
    )
    tune_metric: str = field(
        default="eval_f1",
        metadata={"help": "Metric used for the tuning strategy."},
    )
    perf_tol: Optional[float] = field(
        default=None,
        metadata={"help": "Performance tolerance when optimizing the model."},
    )
    verify_loading: bool = field(
        default=False,
        metadata={"help": "Whether or not to verify the loading of the quantized model."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, OptimizationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, optim_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, optim_args = parser.parse_args_into_dataclasses()

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

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

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
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

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

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
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

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    metric = load_metric("seqeval")

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

    # Initialize our Trainer
    trainer = IncTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    eval_dataloader = trainer.get_eval_dataloader()
    it = iter(eval_dataloader)
    try:
        input_names = next(it).keys()
    except StopIteration:
        input_names = None
        logger.warning(
            "Unable to determine the names of the inputs of the model to trace, input_names is set to None and "
            "model.dummy_inputs().keys() will be used instead."
        )

    resume_from_checkpoint = training_args.resume_from_checkpoint
    metric_name = optim_args.tune_metric

    def take_eval_steps(model, trainer, metric_name, save_metrics=False):
        trainer.model = model
        metrics = trainer.evaluate()
        if save_metrics:
            trainer.save_metrics("eval", metrics)
        logger.info("{}: {}".format(metric_name, metrics.get(metric_name)))
        logger.info("Throughput: {} samples/sec".format(metrics.get("eval_samples_per_second")))
        return metrics.get(metric_name)

    def eval_func(model):
        return take_eval_steps(model, trainer, metric_name)

    def take_train_steps(model, trainer, resume_from_checkpoint, last_checkpoint):
        trainer.model_wrapped = model
        trainer.model = model
        checkpoint = None
        if resume_from_checkpoint is not None:
            checkpoint = resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(pruner, resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def train_func(model):
        return take_train_steps(model, trainer, resume_from_checkpoint, last_checkpoint)

    quantizer = None
    pruner = None

    if not optim_args.quantize and not optim_args.prune:
        raise ValueError("quantize and prune are both set to False.")

    result_baseline_model = take_eval_steps(model, trainer, metric_name)

    default_config = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)), "config")

    if optim_args.quantize:

        from optimum.intel.neural_compressor import IncQuantizationConfig, IncQuantizationMode, IncQuantizer

        if not training_args.do_eval:
            raise ValueError("do_eval must be set to True for quantization.")

        q8_config = IncQuantizationConfig.from_pretrained(
            optim_args.quantization_config if optim_args.quantization_config is not None else default_config,
            config_file_name="quantization.yml",
            cache_dir=model_args.cache_dir,
        )

        # Set metric tolerance if specified
        if optim_args.perf_tol is not None:
            q8_config.set_tolerance(optim_args.perf_tol)

        # Set quantization approach if specified
        if optim_args.quantization_approach is not None:
            supported_approach = {"static", "dynamic", "aware_training"}
            if optim_args.quantization_approach not in supported_approach:
                raise ValueError(
                    "Unknown quantization approach. Supported approach are " + ", ".join(supported_approach)
                )
            quant_approach = getattr(IncQuantizationMode, optim_args.quantization_approach.upper()).value
            q8_config.set_config("quantization.approach", quant_approach)

        # torch FX used for post-training quantization and quantization aware training
        # dynamic quantization will be added when torch FX is more mature
        if q8_config.get_config("quantization.approach") != IncQuantizationMode.DYNAMIC.value:
            from transformers.utils.fx import symbolic_trace

            if not training_args.do_train:
                raise ValueError("do_train must be set to True for static and aware training quantization.")

            # TODO : Remove when dynamic axes support
            if (
                not training_args.dataloader_drop_last
                and eval_dataset.shape[0] % training_args.per_device_eval_batch_size != 0
            ):
                raise ValueError(
                    "The number of samples of the dataset is not a multiple of the batch size."
                    "Use --dataloader_drop_last to overcome."
                )
            if not data_args.pad_to_max_length:
                raise ValueError(
                    "All the samples must have the same sequence length, use --pad_to_max_length to overcome."
                )

            q8_config.set_config("model.framework", "pytorch_fx")
            model.config.save_pretrained(training_args.output_dir)
            model = symbolic_trace(
                model,
                input_names=input_names,
                batch_size=training_args.per_device_eval_batch_size,
                sequence_length=data_args.max_seq_length,
            )

        calib_dataloader = trainer.get_train_dataloader()
        inc_quantizer = IncQuantizer(
            model, q8_config, eval_func=eval_func, train_func=train_func, calib_dataloader=calib_dataloader
        )
        quantizer = inc_quantizer.fit()

    if optim_args.prune:

        from optimum.intel.neural_compressor import IncPruner, IncPruningConfig

        if not training_args.do_train:
            raise ValueError("do_train must be set to True for pruning.")

        pruning_config = IncPruningConfig.from_pretrained(
            optim_args.pruning_config if optim_args.pruning_config is not None else default_config,
            config_file_name="prune.yml",
            cache_dir=model_args.cache_dir,
        )

        # Set targeted sparsity if specified
        if optim_args.target_sparsity is not None:
            pruning_config.set_config("pruning.approach.weight_compression.target_sparsity", optim_args.target_sparsity)

        pruning_start_epoch = pruning_config.get_config("pruning.approach.weight_compression.start_epoch")
        pruning_end_epoch = pruning_config.get_config("pruning.approach.weight_compression.end_epoch")

        if pruning_start_epoch > training_args.num_train_epochs - 1:
            logger.warning(
                f"Pruning end epoch {pruning_start_epoch} is higher than the total number of training epoch "
                f"{training_args.num_train_epochs}. No pruning will be applied."
            )

        if pruning_end_epoch > training_args.num_train_epochs - 1:
            logger.warning(
                f"Pruning end epoch {pruning_end_epoch} is higher than the total number of training epoch "
                f"{training_args.num_train_epochs}. The target sparsity will not be reached."
            )

        inc_pruner = IncPruner(model, pruning_config, eval_func=eval_func, train_func=train_func)

        # Creation Pruning object used for IncTrainer training loop
        pruner = inc_pruner.fit()

    from neural_compressor.experimental import common
    from neural_compressor.experimental.scheduler import Scheduler

    scheduler = Scheduler()
    scheduler.model = common.Model(model)

    if pruner is not None:
        scheduler.append(pruner)

    if quantizer is not None:
        scheduler.append(quantizer)

    opt_model = scheduler()

    _, sparsity = opt_model.report_sparsity()
    result_opt_model = take_eval_steps(opt_model.model, trainer, metric_name, save_metrics=True)

    trainer.save_model(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, CONFIG_NAME), "w") as f:
        yaml.dump(opt_model.tune_cfg, f, default_flow_style=False)

    logger.info(
        f"Optimized model with final sparsity of {sparsity} and {metric_name} of {result_opt_model} saved to: "
        f"{training_args.output_dir}. Original model had an {metric_name} of {result_baseline_model}"
    )

    if optim_args.quantize and optim_args.verify_loading:
        from optimum.intel.neural_compressor.quantization import IncQuantizedModelForTokenClassification

        # Load the model obtained after Intel Neural Compressor (INC) quantization
        loaded_model = IncQuantizedModelForTokenClassification.from_pretrained(
            training_args.output_dir,
            input_names=input_names,
            batch_size=training_args.per_device_eval_batch_size,
            sequence_length=data_args.max_seq_length,
        )
        loaded_model.eval()
        result_loaded_model = take_eval_steps(loaded_model, trainer, metric_name)

        if result_loaded_model != result_opt_model:
            raise ValueError("The quantized model was not successfully loaded.")
        else:
            logger.info(f"The quantized model was successfully loaded.")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
