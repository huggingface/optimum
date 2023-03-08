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

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from transformers.utils import is_tf_available

from optimum.utils.preprocessing import TaskProcessorsManager
from optimum.utils.save_utils import maybe_load_preprocessors

from ...utils import logging
from ..error_utils import AtolError, OutputMatchError, ShapeError


if TYPE_CHECKING:
    if is_tf_available():
        from transformers import PretrainedConfig, TFPreTrainedModel
    from .base import TFLiteConfig


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


def create_calibration_dataset(
    task: str,
    config: "PretrainedConfig",
    model_signatures,
    dataset_name_or_path: Union[str, Path],
    preprocessor_name_or_path: Union[str, Path],
    num_calibration_samples: int = 200,
    calibration_split: str = "train",
    data_keys: Dict[str, Optional[str]] = None,
):
    preprocessor = maybe_load_preprocessors(preprocessor_name_or_path)[0]
    TaskProcessorsManager.get_dataset_processing_class_for_task(task)
    dataset_processing = TaskProcessorsManager.for_task(
        task,
        config=config,
        dataset_path=dataset_name_or_path,
        preprocessor=preprocessor,
        num_calibration_samples=num_calibration_samples,
        calibration_split=calibration_split,
        static_quantization=True,
        data_keys={"primary": primary_key_name, "secondary": secondary_key_name},
    )
    dataset = dataset_processing.load_datasets()["calibration"]

    def calibration_dataset():
        for data in dataset:
            processed_data = {}
            for signature_name, function in model_signatures.items():
                input_names = set(input_.name for input_ in function.input_signature)
                processed_data[signature_name] = {k: v for k, v in data.items() if k in input_names}
            yield processed_data

    return calibration_dataset


def export(
    model: "TFPreTrainedModel",
    config: "TFLiteConfig",
    output: Path,
    quantization: Optional[str] = None,
    calibration_dataset: Optional[Union[str, Path]] = None,
    num_calibration_samples: int = 200,
    calibration_split: str = "train",
    primary_key_name: Optional[str] = None,
    secondary_key_name: Optional[str] = None,
    preprocessor_name_or_path: Optional[Union[str, Path]] = None,
    fallback_to_float: Optional[bool] = True,
    inputs_dtype: Optional[str] = None,
    outputs_dtype: Optional[str] = None,
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

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the TFLite configuration.
    """
    if not is_tf_available():
        raise ImportError("Cannot convert because TensorFlow is not installed. " "Please install TensorFlow first.")
    import tensorflow as tf

    # TODO: validate quantization argument values.

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

        if quantization in ["int8", "int8x16"]:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if calibration_dataset is None:
                logger.info(
                    "Performing dynamic quantization because no calibration dataset was provided, specify one to perform "
                    "static quantization."
                )
            else:
                if preprocessor_name_or_path is None:
                    raise ValueError(
                        "A processor name or path needs to be provided when providing a calibration dataset."
                    )
                converter.representative_dataset = create_calibration_dataset(
                    config.task,
                    model.config,
                    signatures,
                    calibration_dataset,
                    preprocessor_name_or_path,
                    num_calibration_samples=num_calibration_samples,
                    calibration_split=calibration_split,
                    primary_key_name=primary_key_name,
                    secondary_key_name=secondary_key_name,
                )
            if quantization == "int8":
                if calibration_dataset is not None:
                    opsset = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                else:
                    opsset = []
            else:
                logger.warning(
                    "The latency with 8x16 quantization can be much slower than int8 only because it is currently an "
                    "experimental feature, use this only if necessary."
                )
                opsset = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            if fallback_to_float:
                opsset.append(tf.lite.OpsSet.TFLITE_BUILTINS)
            converter.target_spec.supported_ops = opsset
            if inputs_dtype is not None:
                converter.inference_input_type = str_to_dtype[inputs_dtype]
            if outputs_dtype is not None:
                converter.inference_output_type = str_to_dtype[outputs_dtype]
        elif quantization == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

    with open(output, "wb") as fp:
        fp.write(tflite_model)

    return config.inputs, config.outputs
