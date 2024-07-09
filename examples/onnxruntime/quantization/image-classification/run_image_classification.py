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

""" Finetuning the library models for image classification on beans."""
# You can also adapt this script on your own image classification task. Pointers for this are left as comments.
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from evaluate import load
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import AutoFeatureExtractor, EvalPrediction, HfArgumentParser, TrainingArguments
from transformers.utils.versions import require_version

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, QuantizationConfig
from optimum.onnxruntime.modeling_ort import ORTModelForImageClassification
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor
from optimum.onnxruntime.preprocessors.passes import (
    ExcludeGeLUNodes,
    ExcludeLayerNormNodes,
    ExcludeNodeAfter,
    ExcludeNodeFollowedBy,
)
from optimum.onnxruntime.utils import evaluation_loop


logger = logging.getLogger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
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
    train_dir: Optional[str] = field(default=None, metadata={"help": "A directory path for the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A directory path for the validation data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


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

    # Sanity checks
    if data_args.dataset_name is None and data_args.train_dir is None and data_args.validation_dir is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(data_args.dataset_name)
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            task="image-classification",
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.

    labels_column = (
        "labels" if "labels" in dataset["validation"].column_names else dataset["validation"].column_names[1]
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_function(example_batch):
        """Apply transforms across a batch."""
        example_batch["pixel_values"] = [
            transforms(image.convert("RGB")).to(torch.float32).numpy() for image in example_batch["image"]
        ]
        return example_batch

    metric = load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        result = metric.compute(predictions=preds, references=p.label_ids)
        return result

    # Export the model
    model = ORTModelForImageClassification.from_pretrained(model_args.model_name_or_path, export=True)

    # Create the quantizer
    quantizer = ORTQuantizer.from_pretrained(model)

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

    ranges = None
    # Create a quantization preprocessor to determine the nodes to exclude
    quantization_preprocessor = QuantizationPreprocessor()
    if apply_static_quantization:
        calibration_dataset = dataset["train"]
        if optim_args.num_calibration_samples is not None:
            calibration_dataset = calibration_dataset.shuffle(seed=training_args.seed).select(
                range(optim_args.num_calibration_samples)
            )

        # all images are loaded in memory, which could prove expensive if num_calibration_samples is large
        calibration_dataset = calibration_dataset.map(
            partial(preprocess_function),
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running preprocessing on calibration dataset",
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
    model = ORTModelForImageClassification.from_pretrained(
        quantized_model_path, provider=optim_args.execution_provider
    )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))

        try:
            eval_dataset = eval_dataset.align_labels_with_mapping(
                label2id=model.config.label2id, label_column=labels_column
            )
        except Exception:
            logger.warning(
                f"\nModel label mapping: {model.config.label2id}"
                f"\nDataset label features: {eval_dataset.features[labels_column]}"
                f"\nCould not guarantee the model label mapping and the dataset labels match."
                f" Evaluation results may suffer from a wrong matching."
            )

        # Set the validation transforms
        eval_dataset = eval_dataset.with_transform(preprocess_function)

        outputs = evaluation_loop(
            model=model,
            dataset=eval_dataset,
            label_names=[labels_column],
            compute_metrics=compute_metrics,
        )
        # Save metrics
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(outputs.metrics, f, indent=4, sort_keys=True)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
