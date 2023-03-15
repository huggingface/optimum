# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""TensorFlow Lite model check and export functions."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from optimum.onnxruntime.configuration import QuantizationConfig
from transformers import PreTrainedTokenizerBase
from transformers.utils import is_tf_available

from ...utils import logging
from ...utils.preprocessing import Preprocessor, TaskProcessorsManager
from ..error_utils import AtolError, OutputMatchError, ShapeError
from .base import QuantizationApproach


if TYPE_CHECKING:
    from datasets import Dataset

    if is_tf_available():
        from transformers import TFPreTrainedModel
        import tensorflow as tf
    from .base import TFLiteConfig


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class QuantizationConfig:
    quantization: Optional[Union[str, QuantizationApproach]] = None
    fallback_to_float: bool = False
    inputs_dtype: Optional[str] = None
    outputs_dtype: Optional[str] = None
    calibration_dataset_name_or_path: Optional[Union[str, Path]] = None
    calibration_dataset_config_name: Optional[str] = None
    preprocessor: Optional[Preprocessor] = None
    num_calibration_samples: int = 200
    calibration_split: Optional[str] = None
    primary_key: Optional[str] = None
    secondary_key: Optional[str] = None
    question_key: Optional[str] = None
    context_key: Optional[str] = None
    image_key: Optional[str] = None



def validate_model_outputs(
    config: "TFLiteConfig",
    reference_model: "TFPreTrainedModel",
    tflite_model_path: Path,
    tflite_named_outputs: List[str],
    atol: Optional[float] = None,
):
    """
    Validates the export by checking that the outputs from both the reference and the exported model match.

    Args:
        config ([`~optimum.exporters.tflite.TFLiteConfig`]:
            The configuration used to export the model.
        reference_model ([`~TFPreTrainedModel`]):
            The model used for the export.
        tflite_model_path (`Path`):
            The path to the exported model.
        tflite_named_outputs (`List[str]`):
            The names of the outputs to check.
        atol (`Optional[float]`, defaults to `None`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    if not is_tf_available():
        raise ImportError(
            "Cannot validate conversion because TensorFlow is not installed. " "Please install TensorFlow first."
        )
    import tensorflow as tf

    logger.info("Validating TFLite model...")

    if atol is None:
        if isinstance(config.ATOL_FOR_VALIDATION, dict):
            atol = config.ATOL_FOR_VALIDATION[config.task]
        else:
            atol = config.ATOL_FOR_VALIDATION

    inputs = config.generate_dummy_inputs()

    ref_outputs = reference_model(**inputs)

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path.as_posix())
    tflite_model_runner = interpreter.get_signature_runner("model")
    tflite_outputs = tflite_model_runner(**inputs)

    # Check we have a subset of the keys into onnx_outputs against ref_outputs
    ref_outputs_set, tflite_output_set = set(ref_outputs.keys()), set(tflite_named_outputs)
    if not tflite_output_set.issubset(ref_outputs_set):
        raise OutputMatchError(
            "TFLite model output names do not match reference model output names.\n"
            f"Reference model output names: {ref_outputs_set}\n"
            f"TFLite model output names: {tflite_output_set}"
            f"Difference: {tflite_output_set.difference(ref_outputs_set)}"
        )
    else:
        tflite_output_names = ", ".join(tflite_output_set)
        logger.info(f"\t-[✓] TFLite model output names match reference model ({tflite_output_names})")

    # Check the shape and values match
    shape_failures = []
    value_failures = []
    for name, output in tflite_outputs.items():
        if name not in tflite_output_set:
            continue

        ref_output = ref_outputs[name].numpy()

        logger.info(f'\t- Validating TFLite Model output "{name}":')

        # Shape
        if not output.shape == ref_output.shape:
            logger.error(f"\t\t-[x] shape {output.shape} doesn't match {ref_output.shape}")
            shape_failures.append((name, ref_output.shape, output.shape))
        else:
            logger.info(f"\t\t-[✓] {output.shape} matches {ref_output.shape}")

        # Values
        if not np.allclose(ref_output, output, atol=atol):
            max_diff = np.amax(np.abs(ref_output - output))
            logger.error(f"\t\t-[x] values not close enough, max diff: {max_diff} (atol: {atol})")
            value_failures.append((name, max_diff))
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")

    if shape_failures:
        msg = "\n".join(f"- {t[0]}: got {t[1]} (reference) and {t[2]} (TFLite)" for t in shape_failures)
        raise ShapeError("Output shapes do not match between reference model and the TFLite exported model:\n" "{msg}")

    if value_failures:
        msg = "\n".join(f"- {t[0]}: max diff = {t[1]}" for t in value_failures)
        raise AtolError(
            "The maximum absolute difference between the output of the reference model and the TFLite "
            f"exported model is not within the set tolerance {atol}:\n{msg}"
        )


def create_representative_dataset(signatures, dataset: "Dataset"):
    def representative_dataset():
        for sig_name, tf_function in signatures.items():
            inputs_to_keep = None
            for example in dataset:
                if inputs_to_keep is None:
                    args, kwargs = tf_function.structured_input_signature
                    args_to_keep = {input_.name for input_ in args if input_.name in example}
                    kwargs_to_keep = {input_.name for input_ in kwargs.values() if input_.name in example}
                    inputs_to_keep = args_to_keep | kwargs_to_keep
                yield sig_name, {name: value for name, value in example.items() if name in inputs_to_keep}

    return representative_dataset


def prepare_converter_for_quantization(converter: "tf.lite.TFLiteConverter", quantization_config: QuantizationConfig):


def export(
    model: "TFPreTrainedModel",
    config: "TFLiteConfig",
    output: Path,
    quantization: Optional[Union[str, QuantizationApproach]] = None,
    fallback_to_float: bool = False,
    inputs_dtype: Optional[str] = None,
    outputs_dtype: Optional[str] = None,
    calibration_dataset_name_or_path: Optional[Union[str, Path]] = None,
    calibration_dataset_config_name: Optional[str] = None,
    preprocessor: Optional[Preprocessor] = None,
    num_calibration_samples: int = 200,
    calibration_split: Optional[str] = None,
    primary_key: Optional[str] = None,
    secondary_key: Optional[str] = None,
    question_key: Optional[str] = None,
    context_key: Optional[str] = None,
    image_key: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a TensorFlow model to a TensorFlow Lite model.

    Args:
        model ([`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.tflite.TFLiteConfig`]):
            The TFLite configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported TFLite model.
        quantization (`Optional[Union[str, QuantizationApproach]]`, defaults to `None`):
            The quantization to perform. No quantization is applied if left unspecified.
        fallback_to_float (`bool`, defaults to `False`):
            Allows to fallback to float kernels in quantization.
        inputs_dtype (`Optional[str]`, defaults to `None`):
            The data type of the inputs. If specified it must be either "int8" or "uint8". It allows to always take
            integers as inputs, it is useful for interger-only hardware.
        outputs_dtype (`Optional[str]`, defaults to `None`):
            The data type of the outputs. If specified it must be either "int8" or "uint8". It allows to always output
            integers, it is useful for interger-only hardware.
        calibration_dataset_name_or_path (`Optional[Union[str, Path]]`, defaults to `None`):
            The dataset to use for calibrating the quantization parameters for static quantization. If left unspecified,
            a default dataset for the considered task will be used.
        calibration_dataset_config_name (`Optional[str]`, defaults to `None`):
            The configuration name of the dataset if needed.
        preprocessor (`Optional[Preprocessor]`, defaults to `None`):
            The preprocessor associated to the model. This is used for preprocessing the dataset before feeding data to
            the model during calibration.
        num_calibration_samples (`int`, defaults to `200`):
            The number of example from the calibration dataset to use to compute the quantization parameters.
        calibration_split (`Optional[str]`, defaults to `None`):
            The split of the dataset to use. If none is specified and the dataset contains multiple splits, the
            smallest split will be used.
        primary_key (`Optional[str]`, defaults `None`):
            The name of the column in the dataset containing the main data to preprocess. Only for
            sequence-classification and token-classification.
        secondary_key (`Optional[str]`, defaults `None`):
            The name of the second column in the dataset containing the main data to preprocess, not always needed.
            Only for sequence-classification and token-classification.
        question_key (`Optional[str]`, defaults `None`):
            The name of the column containing the question in the dataset. Only for question-answering.
        context_key (`Optional[str]`, defaults `None`):
            The name of the column containing the context in the dataset. Only for question-answering.
        image_key (`Optional[str]`, defaults `None`):
            The name of the column containing the image in the dataset. Only for image-classification.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the TFLite configuration.
    """
    if not is_tf_available():
        raise ImportError("Cannot convert because TensorFlow is not installed. " "Please install TensorFlow first.")
    import tensorflow as tf

    str_to_dtype = {"int8": tf.int8, "uint8": tf.uint8}

    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using TensorFlow: {tf.__version__}")
    model.config.return_dict = True

    # Check if we need to override certain configuration item
    if config.values_override is not None:
        logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in config.values_override.items():
            logger.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    signatures = config.model_to_signatures(model)

    with TemporaryDirectory() as tmp_dir_name:
        model.save(tmp_dir_name, signatures=signatures)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir_name)

        if quantization is not None:
            if isinstance(quantization, str) and not isinstance(quantization, QuantizationApproach):
                quantization = QuantizationApproach(quantization)

        if quantization in [QuantizationApproach.INT8, QuantizationApproach.INT8x16]:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Handling the calibration dataset:
            # - Either loading the default dataset if no calibration dataset was provided or the required dataset,
            # - Splitting the dataset with the provided a dataset split or with the first split if none is provided.
            # - Shuffling the split.
            # - Selecting num_calibration_samples in the dataset split.
            # - Batching the dataset.
            # - Converting it to the TensorFlow format.
            from ...exporters import TasksManager

            task = TasksManager.get_task_from_model(model)
            preprocessor_kwargs = {}
            if isinstance(preprocessor, PreTrainedTokenizerBase):
                preprocessor_kwargs["max_length"] = config.sequence_length
            task_processor = TaskProcessorsManager.get_task_processor_class_for_task(task)(
                model.config, preprocessor, preprocessor_kwargs
            )

            load_smallest_split = calibration_split is None
            if load_smallest_split:
                logger.warning(
                    "Since no calibration split was provided for the calibration dataset, the smallest split will be "
                    "used if the dataset contains multiple splits."
                )

            if calibration_dataset_name_or_path is None:
                calibration_dataset = task_processor.load_default_dataset(
                    only_keep_necessary_columns=True, load_smallest_split=load_smallest_split, split=calibration_split
                )
            else:
                data_keys = {}
                if primary_key is not None:
                    data_keys["primary"] = primary_key
                if secondary_key is not None:
                    data_keys["secondary"] = secondary_key
                if question_key is not None:
                    data_keys["question"] = question_key
                if context_key is not None:
                    data_keys["context"] = context_key
                if image_key is not None:
                    data_keys["imagey"] = image_key

                calibration_dataset = task_processor.load_dataset(
                    calibration_dataset_name_or_path,
                    data_keys=data_keys,
                    only_keep_necessary_columns=True,
                    load_smallest_split=load_smallest_split,
                    name=calibration_dataset_config_name,
                    split=calibration_split,
                )

            calibration_dataset = calibration_dataset.shuffle()

            batch_size = config.batch_size
            if num_calibration_samples % batch_size != 0:
                new_num_calibration_samples = (num_calibration_samples // batch_size + 1) * batch_size
                logger.info(
                    f"The number of calibration examples ({num_calibration_samples}) does not divide the batch size "
                    "({batch_size}), using {new_num_calibration_samples} examples instead."
                )
                num_calibration_samples = new_num_calibration_samples

            if num_calibration_samples > calibration_dataset.num_rows:
                raise ValueError(
                    f"There are only {calibration_dataset.num_rows} examples in the calibration dataset, but it was "
                    "requested to perform calibration using {num_calibration_samples} examples."
                )

            calibration_dataset = calibration_dataset.select(range(num_calibration_samples))

            if batch_size > 1:

                def batching_function(examples):
                    return {column_name: [list(examples[column_name])] for column_name in examples.keys()}

                calibration_dataset = calibration_dataset.map(batching_function, batched=True, batch_size=batch_size)

            calibration_dataset = calibration_dataset.with_format("tf")
            converter.representative_dataset = create_representative_dataset(signatures, calibration_dataset)

            # Handling the OpsSet.
            if quantization is QuantizationApproach.INT8:
                opsset = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            else:
                logger.warning(
                    "The latency with 8x16 quantization can be much slower than int8 only because it is currently an "
                    "experimental feature, use this only if necessary."
                )
                opsset = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            if fallback_to_float:
                opsset.append(tf.lite.OpsSet.TFLITE_BUILTINS)
            converter.target_spec.supported_ops = opsset

            # Handling the inputs and outputs dtype, this allows to have a TFLite model taking integers inputs and
            # outputting integers outputs, needed for integer-only hardware.
            if inputs_dtype is not None:
                converter.inference_input_type = str_to_dtype[inputs_dtype]
            if outputs_dtype is not None:
                converter.inference_output_type = str_to_dtype[outputs_dtype]
        elif quantization is QuantizationApproach.INT8_DYNAMIC:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization is QuantizationApproach.FP16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

    with open(output, "wb") as fp:
        fp.write(tflite_model)

    return config.inputs, config.outputs
