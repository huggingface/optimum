#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.fx import symbolic_trace
from transformers.utils.versions import require_version

import yaml
from optimum.intel.neural_compressor import (
    IncOptimizer,
    IncPruner,
    IncPruningConfig,
    IncQuantizationConfig,
    IncQuantizationMode,
    IncQuantizer,
    IncTrainer,
)
from optimum.intel.neural_compressor.quantization import IncQuantizedModelForCausalLM
from optimum.intel.neural_compressor.utils import CONFIG_NAME


os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/inc/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
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

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


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
        default="eval_loss",
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

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


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
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Initialize our Trainer
    trainer = IncTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
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
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
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
        return trainer.model

    def train_func(model):
        return take_train_steps(model, trainer, resume_from_checkpoint, last_checkpoint)

    quantizer = None
    pruner = None

    if not optim_args.quantize and not optim_args.prune:
        raise ValueError("quantize and prune are both set to False.")

    result_baseline_model = take_eval_steps(model, trainer, metric_name)

    default_config = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)), "config")

    if optim_args.quantize:

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

        q8_config.set_config("tuning.accuracy_criterion.higher_is_better", False)

        quant_approach = IncQuantizationMode(q8_config.get_config("quantization.approach"))
        # torch FX used for post-training quantization and quantization aware training
        # dynamic quantization will be added when torch FX is more mature
        if quant_approach != IncQuantizationMode.DYNAMIC:
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

            q8_config.set_config("model.framework", "pytorch_fx")
            model.config.save_pretrained(training_args.output_dir)
            model = symbolic_trace(
                model,
                input_names=input_names,
                batch_size=training_args.per_device_eval_batch_size,
                sequence_length=block_size,
            )

        calib_dataloader = trainer.get_train_dataloader() if quant_approach == IncQuantizationMode.STATIC else None
        inc_quantizer = IncQuantizer(
            model, q8_config, eval_func=eval_func, train_func=train_func, calib_dataloader=calib_dataloader
        )
        quantizer = inc_quantizer.fit()

    if optim_args.prune:

        if not training_args.do_train:
            raise ValueError("do_train must be set to True for pruning.")

        pruning_config = IncPruningConfig.from_pretrained(
            optim_args.pruning_config if optim_args.pruning_config is not None else default_config,
            config_file_name="prune.yml",
            cache_dir=model_args.cache_dir,
        )

        # Set targeted sparsity if specified
        if optim_args.target_sparsity is not None:
            pruning_config.set_config(
                "pruning.approach.weight_compression.target_sparsity", optim_args.target_sparsity
            )

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

    inc_optimizer = IncOptimizer(model, quantizer=quantizer, pruner=pruner)
    opt_model = inc_optimizer.fit()
    result_opt_model = take_eval_steps(opt_model.model, trainer, metric_name, save_metrics=True)

    trainer.save_model(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, CONFIG_NAME), "w") as f:
        yaml.dump(opt_model.tune_cfg, f, default_flow_style=False)

    logger.info(
        f"Optimized model with {metric_name} of {result_opt_model} saved to: {training_args.output_dir}."
        f" Original model had an {metric_name} of {result_baseline_model}."
    )

    if optim_args.quantize and optim_args.verify_loading:

        # Load the model obtained after Intel Neural Compressor (INC) quantization
        loaded_model = IncQuantizedModelForCausalLM.from_pretrained(
            training_args.output_dir,
            input_names=input_names,
            batch_size=training_args.per_device_eval_batch_size,
            sequence_length=block_size,
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
