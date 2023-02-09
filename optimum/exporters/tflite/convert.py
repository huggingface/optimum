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
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from transformers.utils import is_tf_available

from ...utils import logging
from ..error_utils import AtolError, OutputMatchError, ShapeError


if TYPE_CHECKING:
    if is_tf_available():
        from transformers import TFPreTrainedModel
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


def export(
    model: "TFPreTrainedModel",
    config: "TFLiteConfig",
    output: Path,
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
        tflite_model = converter.convert()

    with open(output, "wb") as fp:
        fp.write(tflite_model)

    return config.inputs, config.outputs
