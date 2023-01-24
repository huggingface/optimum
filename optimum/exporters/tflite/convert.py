# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from transformers.utils import is_tf_available

from ...utils import logging


if TYPE_CHECKING:
    if is_tf_available():
        from transformers import TFPreTrainedModel
    from .base import TFLiteConfig


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ShapeError(ValueError):
    pass


class AtolError(ValueError):
    pass


class OutputMatchError(ValueError):
    pass


# def check_dummy_inputs_are_allowed(
#     model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], dummy_input_names: Iterable[str]
# ):
#     """
#     Checks that the dummy inputs from the ONNX config is a subset of the allowed inputs for `model`.
#     Args:
#         model (`Union[transformers.PreTrainedModel, transformers.TFPreTrainedModel`]):
#             The model instance.
#         model_inputs (`Iterable[str]`):
#             The model input names.
#     """
#
#     forward = model.forward if is_torch_available() and isinstance(model, nn.Module) else model.call
#     forward_parameters = signature(forward).parameters
#     forward_inputs_set = set(forward_parameters.keys())
#     dummy_input_names = set(dummy_input_names)
#
#     # We are fine if config_inputs has more keys than model_inputs
#     if not dummy_input_names.issubset(forward_inputs_set):
#         raise ValueError(
#             f"Config dummy inputs are not a subset of the model inputs: {dummy_input_names} vs {forward_inputs_set}"
#         )


def validate_model_outputs(
    config: "TFLiteConfig",
    reference_model: "TFPreTrainedModel",
    tflite_model_path: Path,
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

    logger.info("Validating TensorFlow Lite model...")

    if atol is None:
        if isinstance(config.ATOL_FOR_VALIDATION, dict):
            atol = config.ATOL_FOR_VALIDATION[config.task]
        else:
            atol = config.ATOL_FOR_VALIDATION

    inputs = config.generate_dummy_inputs()

    ref_outputs = reference_model(**inputs)
    # ref_outputs_dict = {}

    # We flatten potential collection of outputs (i.e. past_keys) to a flat structure
    # for name, value in ref_outputs.items():
    #     # Overwriting the output name as "present" since it is the name used for the ONNX outputs
    #     # ("past_key_values" being taken for the ONNX inputs)
    #     if name == "past_key_values":
    #         name = "present"
    #     if isinstance(value, (list, tuple)):
    #         value = config.flatten_output_collection_property(name, value)
    #         ref_outputs_dict.update(value)
    #     else:
    #         ref_outputs_dict[name] = value

    # Create onnxruntime inputs from the reference model inputs
    # reference_model_inputs_for_validation = config.generate_dummy_inputs_for_validation(reference_model_inputs)

    # We flatten potential collection of inputs (i.e. past_keys)
    # onnx_inputs = {}
    # for name, value in reference_model_inputs_for_validation.items():
    #     if isinstance(value, (list, tuple)):
    #         value = config.flatten_output_collection_property(name, value)
    #         onnx_inputs.update({tensor_name: pt_tensor.cpu().numpy() for tensor_name, pt_tensor in value.items()})
    #     else:
    #         onnx_inputs[name] = value.cpu().numpy()

    # Compute outputs from the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path.as_posix())
    interpreter.allocate_tensors()
    for input_detail in interpreter.get_input_details():
        interpreter.set_tensor(input_detail["index"], inputs[input_detail["name"]])
    tflite_outputs = [interpreter.get_tensor(output["index"]) for output in interpreter.get_output_details()]

    # TODO: enable that once able to export the output names.
    # Check we have a subset of the keys into onnx_outputs against ref_outputs
    # ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
    # if not onnx_outputs_set.issubset(ref_outputs_set):
    #     raise OutputMatchError(
    #         "ONNX model output names do not match reference model output names.\n"
    #         f"Reference model output names: {ref_outputs_set}\n"
    #         f"ONNX model output names: {onnx_outputs_set}"
    #         f"Difference: {onnx_outputs_set.difference(ref_outputs_set)}"
    #     )
    # else:
    #     onnx_output_names = ", ".join(onnx_outputs_set)
    #     logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_output_names})")

    # Check the shape and values match
    shape_failures = []
    value_failures = []
    for name, output in zip(config.outputs, tflite_outputs):
        ref_output = ref_outputs[name].numpy()

        logger.info(f'\t- Validating ONNX Model output "{name}":')

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
        msg = "\n".join(f"- {t[0]}: got {t[1]} (reference) and {t[2]} (ONNX)" for t in shape_failures)
        raise ShapeError(
            f"Output shapes do not match between reference model and TensorFlow Lite exported model:\n" "{msg}"
        )

    if value_failures:
        msg = "\n".join(f"- {t[0]}: max diff = {t[1]}" for t in value_failures)
        raise AtolError(
            "The maximum absolute difference between the output of the reference model and the TensorFlow Lite "
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
            The ONNX configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported ONNX model.
        opset (`Optional[int]`, defaults to `None`):
            The version of the ONNX operator set to use.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    if not is_tf_available():
        raise ImportError("Cannot convert because TensorFlow is not installed. " "Please install TensorFlow first.")
    import tensorflow as tf

    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using TensorFlow: {tf.__version__}")
    model.config.return_dict = True

    # TODO: enable config override.
    # Check if we need to override certain configuration item
    # if config.values_override is not None:
    #     logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
    #     for override_config_key, override_config_value in config.values_override.items():
    #         logger.info(f"\t- {override_config_key} -> {override_config_value}")
    #         setattr(model.config, override_config_key, override_config_value)

    func = config.model_to_tf_function(model, concrete=True)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
    tflite_model = converter.convert()

    with open(output, "wb") as fp:
        fp.write(tflite_model)

    return config.inputs, config.outputs
