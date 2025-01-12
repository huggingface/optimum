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

""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

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
from datasets import load_dataset
from evaluate import load
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, QuantizationConfig
from optimum.onnxruntime.modeling_ort import ORTModelForSequenceClassification
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
    "datasets>=1.8.0", "To fix: pip install -r examples/onnxruntime/quantization/text-classification/requirements.txt"
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


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

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if not is_regression:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    def preprocess_function(examples, tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding="max_length", max_length=min(max_length, tokenizer.model_max_length), truncation=True
        )
        return result

    # Get the metric function
    if data_args.task_name is not None:
        metric = load("glue", data_args.task_name)
    else:
        metric = load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Export the model
    model = ORTModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, export=True)

    # Create the quantizer
    quantizer = ORTQuantizer.from_pretrained(model)

    # Run the tokenizer on the dataset
    preprocessed_datasets = raw_datasets.map(
        partial(preprocess_function, tokenizer=tokenizer, max_length=data_args.max_seq_length),
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

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
        # Create the calibration dataset used for the calibration step
        calibration_dataset = preprocessed_datasets["train"]
        if optim_args.num_calibration_samples is not None:
            calibration_dataset = calibration_dataset.select(range(optim_args.num_calibration_samples))

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
    model = ORTModelForSequenceClassification.from_pretrained(
        quantized_model_path, provider=optim_args.execution_provider
    )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if "validation" not in preprocessed_datasets and "validation_matched" not in preprocessed_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = preprocessed_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        try:
            eval_dataset = eval_dataset.align_labels_with_mapping(label2id=model.config.label2id, label_column="label")
        except Exception:
            logger.warning(
                f"\nModel label mapping: {model.config.label2id}"
                f"\nDataset label features: {eval_dataset.features['label']}"
                f"\nCould not guarantee the model label mapping and the dataset labels match."
                f" Evaluation results may suffer from a wrong matching."
            )

        outputs = evaluation_loop(
            model=model,
            dataset=eval_dataset,
            compute_metrics=compute_metrics,
            label_names=["label"],
        )

        # Save metrics
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(outputs.metrics, f, indent=4, sort_keys=True)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        if "test" not in preprocessed_datasets and "test_matched" not in preprocessed_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = preprocessed_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        outputs = evaluation_loop(
            model=model,
            dataset=predict_dataset,
            label_names=["label"],
        )
        predictions = np.squeeze(outputs.predictions) if is_regression else np.argmax(outputs.predictions, axis=1)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "prediction.txt")
        with open(output_predictions_file, "w") as writer:
            logger.info(f"***** Predict results {data_args.task_name} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if is_regression:
                    writer.write(f"{index}\t{item:3.3f}\n")
                else:
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
