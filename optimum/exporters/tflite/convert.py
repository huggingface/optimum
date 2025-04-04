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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.utils import is_tf_available

from ...utils import logging
from ...utils.preprocessing import Preprocessor, TaskProcessorsManager
from ..error_utils import AtolError, OutputMatchError, ShapeError
from .base import QuantizationApproach, QuantizationApproachNotSupported


if TYPE_CHECKING:
    from datasets import Dataset

    if is_tf_available():
        import tensorflow as tf
        from transformers import TFPreTrainedModel
    from .base import TFLiteConfig, TFLiteQuantizationConfig


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


def prepare_converter_for_quantization(
    model: "TFPreTrainedModel",
    config: "TFLiteConfig",
    preprocessor: Optional[Preprocessor],
    signatures: Dict[str, Callable],
    quantization_config: "TFLiteQuantizationConfig",
    converter: "tf.lite.TFLiteConverter",
    task: Optional[str] = None,
):
    import tensorflow as tf

    if (
        not config.supports_quantization_approach(quantization_config.approach)
        and not quantization_config.fallback_to_float
    ):
        raise QuantizationApproachNotSupported(
            f"{model.config.model_type} do not support full {quantization_config.approach} quantization, use "
            "fallback_to_float=True to fallback to the float implementation for the unsupported ops."
        )

    str_to_dtype = {"int8": tf.int8, "uint8": tf.uint8}
    if quantization_config.approach in [QuantizationApproach.INT8, QuantizationApproach.INT8x16]:
        if preprocessor is None:
            raise ValueError(
                "A preprocessor must be passed for INT8 and INT8x16 quantization since it is needed to preprocess "
                "the calibration dataset."
            )

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Handling the calibration dataset:
        # - Either loading the default dataset if no calibration dataset was provided or the required dataset,
        # - Splitting the dataset with the provided dataset split or with the first split if none is provided.
        # - Shuffling the split.
        # - Selecting num_calibration_samples in the dataset split.
        # - Batching the dataset.
        # - Converting it to the TensorFlow format.

        if task is None:
            from ...exporters import TasksManager

            task = TasksManager.infer_task_from_model(model, library_name="transformers")

        preprocessor_kwargs = {}
        if isinstance(preprocessor, PreTrainedTokenizerBase):
            preprocessor_kwargs["max_length"] = config.sequence_length
        task_processor = TaskProcessorsManager.get_task_processor_class_for_task(task)(
            model.config, preprocessor, preprocessor_kwargs
        )

        if task == "token-classification" and model.config.model_type in {
            "bloom",
            "camembert",
            "deberta",
            "gpt2",
            "roberta",
        }:
            preprocessor.add_prefix_space = True

        load_smallest_split = quantization_config.calibration_split is None
        if load_smallest_split:
            logger.warning(
                "Since no calibration split was provided for the calibration dataset, the smallest split will be "
                "used if the dataset contains multiple splits."
            )

        batch_size = config.batch_size
        num_calibration_samples = quantization_config.num_calibration_samples
        if num_calibration_samples % batch_size != 0:
            new_num_calibration_samples = (num_calibration_samples // batch_size + 1) * batch_size
            logger.info(
                f"The number of calibration examples ({num_calibration_samples}) does not divide the batch size "
                f"({batch_size}), using {new_num_calibration_samples} examples instead."
            )
            num_calibration_samples = new_num_calibration_samples

        if quantization_config.calibration_dataset_name_or_path is None:
            calibration_dataset = task_processor.load_default_dataset(
                only_keep_necessary_columns=True,
                load_smallest_split=load_smallest_split,
                num_samples=num_calibration_samples,
                shuffle=True,
                split=quantization_config.calibration_split,
            )
        else:
            data_keys = {}
            if quantization_config.primary_key is not None:
                data_keys["primary"] = quantization_config.primary_key
            if quantization_config.secondary_key is not None:
                data_keys["secondary"] = quantization_config.secondary_key
            if quantization_config.question_key is not None:
                data_keys["question"] = quantization_config.question_key
            if quantization_config.context_key is not None:
                data_keys["context"] = quantization_config.context_key
            if quantization_config.image_key is not None:
                data_keys["image"] = quantization_config.image_key

            calibration_dataset = task_processor.load_dataset(
                quantization_config.calibration_dataset_name_or_path,
                data_keys=data_keys,
                only_keep_necessary_columns=True,
                load_smallest_split=load_smallest_split,
                num_samples=num_calibration_samples,
                shuffle=True,
                name=quantization_config.calibration_dataset_config_name,
                split=quantization_config.calibration_split,
            )

        if batch_size > 1:
            # Batching can be buggy if a column for on example is empty, and another is not.
            # Since we only care about the columns used by the model (no evaluation), we can filter by only keeping
            # columns that are used at least by one signature. If batching still fails, then it is wanted because it
            # means that there is missing data for calibrating the model.
            columns_needed_by_all_signatures = set()
            for tf_function in signatures.values():
                args, kwargs = tf_function.structured_input_signature
                columns_needed_by_all_signatures |= {input_.name for input_ in args}
                columns_needed_by_all_signatures |= {input_.name for input_ in kwargs.values()}

            # TODO: maybe use calibration.select_columns(columns_needed_by_all_signatures) instead, did not work?
            columns_to_remove = set(calibration_dataset.column_names) - columns_needed_by_all_signatures
            calibration_dataset = calibration_dataset.remove_columns(columns_to_remove)

            def batching_function(examples):
                return {column_name: [examples[column_name]] for column_name in examples.keys()}

            calibration_dataset = calibration_dataset.map(batching_function, batched=True, batch_size=batch_size)

        calibration_dataset = calibration_dataset.with_format("tf")
        converter.representative_dataset = create_representative_dataset(signatures, calibration_dataset)

        # Handling the OpsSet.
        if quantization_config.approach is QuantizationApproach.INT8:
            opsset = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        else:
            logger.warning(
                "The latency with 8x16 quantization can be much slower than int8 only because it is currently an "
                "experimental feature, use this only if necessary."
            )
            opsset = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        if quantization_config.fallback_to_float:
            opsset.append(tf.lite.OpsSet.TFLITE_BUILTINS)
        converter.target_spec.supported_ops = opsset

        # Handling the inputs and outputs dtype, this allows to have a TFLite model taking integers inputs and
        # outputting integers outputs, needed for integer-only hardware.
        if quantization_config.inputs_dtype is not None:
            converter.inference_input_type = str_to_dtype[quantization_config.inputs_dtype]
        if quantization_config.outputs_dtype is not None:
            converter.inference_output_type = str_to_dtype[quantization_config.outputs_dtype]
    elif quantization_config.approach is QuantizationApproach.INT8_DYNAMIC:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization_config.approach is QuantizationApproach.FP16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]


def export(
    model: "TFPreTrainedModel",
    config: "TFLiteConfig",
    output: Path,
    task: Optional[str] = None,
    preprocessor: Optional[Preprocessor] = None,
    quantization_config: Optional["TFLiteQuantizationConfig"] = None,
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
        task (`Optional[str]`, defaults to `None`):
            The task of the model. If left unspecified the task will be inferred automatically. Only needed for static
            quantization.
        preprocessor (`Optional[Preprocessor]`, defaults to `None`):
            The preprocessor associated to the model. This is used for preprocessing the dataset before feeding data to
            the model during calibration.
        quantization_config (`Optional[TFLiteQuantizationConfig]`, defaults to `None`):
            The dataclass containing all the needed information to perform quantization.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the TFLite configuration.
    """
    if not is_tf_available():
        raise ImportError("Cannot convert because TensorFlow is not installed. " "Please install TensorFlow first.")
    import tensorflow as tf

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

        if quantization_config is not None:
            prepare_converter_for_quantization(
                model, config, preprocessor, signatures, quantization_config, converter, task=task
            )

        tflite_model = converter.convert()

    with open(output, "wb") as fp:
        fp.write(tflite_model)

    return config.inputs, config.outputs
