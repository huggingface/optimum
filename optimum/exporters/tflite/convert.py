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
"""ONNX model check and export functions."""

# import os
# from inspect import signature
# from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

import numpy as np
from transformers.utils import is_tf_available, is_torch_available

import onnx

# from ...onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
# from ...utils import TORCH_MINIMUM_VERSION, is_diffusers_available, is_torch_onnx_support_available, logging
# from .utils import recursive_to_device


# if is_torch_available():
#     import torch.nn as nn
#     from transformers.modeling_utils import PreTrainedModel
#     from transformers.pytorch_utils import is_torch_less_than_1_11
# 
# if is_diffusers_available():
#     from diffusers import ModelMixin
# 
# if is_tf_available():
#     from transformers.modeling_tf_utils import TFPreTrainedModel

if TYPE_CHECKING:
    from .base import TFLiteConfig

# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


# def validate_model_outputs(
#     config: TFLiteConfig,
#     reference_model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
#     onnx_model: Path,
#     onnx_named_outputs: List[str],
#     atol: Optional[float] = None,
#     input_shapes: Optional[Dict] = None,
#     device: str = "cpu",
# ):
#     """
#     Validates the export by checking that the outputs from both the reference and the exported model match.
# 
#     Args:
#         config ([`~OnnxConfig`]:
#             The configuration used to export the model.
#         reference_model ([`~PreTrainedModel`] or [`~TFPreTrainedModel`]):
#             The model used for the export.
#         onnx_model (`Path`):
#             The path to the exported model.
#         onnx_named_outputs (`List[str]`):
#             The names of the outputs to check.
#         atol (`Optional[float]`, defaults to `None`):
#             The absolute tolerance in terms of outputs difference between the reference and the exported model.
#         input_shapes (`Optional[Dict]`, defaults to `None`):
#             If specified, allows to use specific shapes to validate the ONNX model on.
#         device (`str`, defaults to `"cpu"`):
#             The device on which the ONNX model will be validated. Either `cpu` or `cuda`. Validation on a CUDA device is supported only for PyTorch.
# 
#     Raises:
#         ValueError: If the outputs shapes or values do not match between the reference and the exported model.
#     """
#     from onnxruntime import InferenceSession, SessionOptions
# 
#     logger.info("Validating ONNX model...")
# 
#     if atol is None:
#         atol = config.ATOL_FOR_VALIDATION
# 
#     if "diffusers" in str(reference_model.__class__) and not is_diffusers_available():
#         raise ImportError("The pip package `diffusers` is required to validate stable diffusion ONNX models.")
# 
#     framework = "pt" if is_torch_available() and isinstance(reference_model, nn.Module) else "tf"
# 
#     if input_shapes is None:
#         input_shapes = {}  # will use the defaults from DEFAULT_DUMMY_SHAPES
#     reference_model_inputs = config.generate_dummy_inputs(framework=framework, **input_shapes)
# 
#     # Create ONNX Runtime session
#     options = SessionOptions()
# 
#     if device.startswith("cuda"):
#         provider = "CUDAExecutionProvider"
#     else:
#         provider = "CPUExecutionProvider"
# 
#     session = InferenceSession(onnx_model.as_posix(), options, providers=[provider])
# 
#     # Compute outputs from the reference model
#     if is_torch_available() and isinstance(reference_model, nn.Module):
#         reference_model.to(device)
# 
#         for key, value in reference_model_inputs.items():
#             reference_model_inputs[key] = recursive_to_device(value=value, device=device)
# 
#     ref_outputs = reference_model(**reference_model_inputs)
#     ref_outputs_dict = {}
# 
#     # We flatten potential collection of outputs (i.e. past_keys) to a flat structure
#     for name, value in ref_outputs.items():
#         # Overwriting the output name as "present" since it is the name used for the ONNX outputs
#         # ("past_key_values" being taken for the ONNX inputs)
#         if name == "past_key_values":
#             name = "present"
#         if isinstance(value, (list, tuple)):
#             value = config.flatten_output_collection_property(name, value)
#             ref_outputs_dict.update(value)
#         else:
#             ref_outputs_dict[name] = value
# 
#     # Create onnxruntime inputs from the reference model inputs
#     reference_model_inputs_for_validation = config.generate_dummy_inputs_for_validation(reference_model_inputs)
# 
#     # We flatten potential collection of inputs (i.e. past_keys)
#     onnx_inputs = {}
#     for name, value in reference_model_inputs_for_validation.items():
#         if isinstance(value, (list, tuple)):
#             value = config.flatten_output_collection_property(name, value)
#             onnx_inputs.update({tensor_name: pt_tensor.cpu().numpy() for tensor_name, pt_tensor in value.items()})
#         else:
#             onnx_inputs[name] = value.cpu().numpy()
# 
#     # Compute outputs from the ONNX model
#     onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)
# 
#     # Modify the ONNX output names to match the reference model output names
#     onnx_named_outputs = config.output_names_for_validation(onnx_named_outputs)
# 
#     # Check we have a subset of the keys into onnx_outputs against ref_outputs
#     ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
#     if not onnx_outputs_set.issubset(ref_outputs_set):
#         raise OutputMatchError(
#             "ONNX model output names do not match reference model output names.\n"
#             f"Reference model output names: {ref_outputs_set}\n"
#             f"ONNX model output names: {onnx_outputs_set}"
#             f"Difference: {onnx_outputs_set.difference(ref_outputs_set)}"
#         )
#     else:
#         onnx_output_names = ", ".join(onnx_outputs_set)
#         logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_output_names})")
# 
#     if "diffusers" in str(reference_model.__class__) and not is_diffusers_available():
#         raise ImportError("The pip package `diffusers` is required to validate stable diffusion ONNX models.")
# 
#     # Check the shape and values match
#     shape_failures = []
#     value_failures = []
#     for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
#         if is_torch_available() and isinstance(reference_model, nn.Module):
#             ref_value = ref_outputs_dict[name].detach().cpu().numpy()
#         else:
#             ref_value = ref_outputs_dict[name].cpu().numpy()
#         logger.info(f'\t- Validating ONNX Model output "{name}":')
# 
#         # Shape
#         if not ort_value.shape == ref_value.shape:
#             logger.error(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
#             shape_failures.append((name, ref_value.shape, ort_value.shape))
#         else:
#             logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")
# 
#         # Values
#         if not np.allclose(ref_value, ort_value, atol=atol):
#             max_diff = np.amax(np.abs(ref_value - ort_value))
#             logger.error(f"\t\t-[x] values not close enough, max diff: {max_diff} (atol: {atol})")
#             value_failures.append((name, max_diff))
#         else:
#             logger.info(f"\t\t-[✓] all values close (atol: {atol})")
# 
#     if shape_failures:
#         msg = "\n".join(f"- {t[0]}: got {t[1]} (reference) and {t[2]} (ONNX)" for t in shape_failures)
#         raise ShapeError(f"Output shapes do not match between reference model and ONNX exported model:\n{msg}")
# 
#     if value_failures:
#         msg = "\n".join(f"- {t[0]}: max diff = {t[1]}" for t in value_failures)
#         raise AtolError(
#             f"The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance {atol}:\n{msg}"
#         )


def export_model(
    model: "TFPreTrainedModel",
    config: "TFLiteConfig",
    opset: int,
    output: Path,
) -> Tuple[List[str], List[str]]:
    """
    Exports a TensorFlow model to an ONNX Intermediate Representation.

    Args:
        model ([`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported ONNX model.
        device (`str`, *optional*, defaults to `cpu`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    # This is needed to import onnx and tf2onnx because onnx is also the name of the current directory.
    import tensorflow as tf

    # logger.info(f"Using framework TensorFlow: {tf.__version__}")

    model.config.return_dict = True

    func = config.model_to_tf_function(model, concrete=True)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
    tflite_model = converter.convert()

    with open(output, "wb") as fp:
        fp.write(tflite_model)

    return config.inputs, config.outputs

    # Check if we need to override certain configuration item
    # TODO: enable config override.
    # if config.values_override is not None:
    #     logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
    #     for override_config_key, override_config_value in config.values_override.items():
    #         logger.info(f"\t- {override_config_key} -> {override_config_value}")
    #         setattr(model.config, override_config_key, override_config_value)

    # Ensure inputs match
    # dummy_inputs = config.generate_dummy_inputs(framework="tf")
    # check_dummy_inputs_are_allowed(model, dummy_inputs)

    # inputs = config.ordered_inputs(model)
    # input_names = list(inputs.keys())
    # output_names = list(config.outputs.keys())

    # config.patch_ops()
    # config.restore_ops()
    

    # return input_names, output_names


# def export(
#     model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
#     config: OnnxConfig,
#     output: Path,
#     opset: Optional[int] = None,
#     device: str = "cpu",
#     input_shapes: Optional[Dict] = None,
# ) -> Tuple[List[str], List[str]]:
#     """
#     Exports a Pytorch or TensorFlow model to an ONNX Intermediate Representation.
# 
#     Args:
#         model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
#             The model to export.
#         config ([`~exporters.onnx.config.OnnxConfig`]):
#             The ONNX configuration associated with the exported model.
#         output (`Path`):
#             Directory to store the exported ONNX model.
#         opset (`Optional[int]`, defaults to `None`):
#             The version of the ONNX operator set to use.
#         device (`str`, *optional*, defaults to `cpu`):
#             The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
#             export on CUDA devices.
#         input_shapes (`Optional[Dict]`, defaults to `None`):
#             If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
# 
#     Returns:
#         `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
#         the ONNX configuration.
#     """
#     if not (is_torch_available() or is_tf_available()):
#         raise ImportError(
#             "Cannot convert because neither PyTorch nor TensorFlow are installed. "
#             "Please install torch or tensorflow first."
#         )
# 
#     output.parent.mkdir(parents=True, exist_ok=True)
# 
#     if opset is None:
#         opset = config.DEFAULT_ONNX_OPSET
# 
#     if "diffusers" in str(model.__class__) and not is_diffusers_available():
#         raise ImportError("The pip package `diffusers` is required to export stable diffusion models to ONNX.")
# 
#     if is_torch_available() and isinstance(model, nn.Module):
#         from ...utils import torch_version
# 
#         if not is_torch_onnx_support_available():
#             raise AssertionError(
#                 f"Unsupported PyTorch version, minimum required is {TORCH_MINIMUM_VERSION}, got: {torch_version}"
#             )
# 
#         if not config.is_torch_support_available:
#             logger.warning(
#                 f"Unsupported PyTorch version for this model. Minimum required is {config.MIN_TORCH_VERSION},"
#                 f" got: {torch.__version__}"
#             )
#         return export_pytorch(model, config, opset, output, device=device, input_shapes=input_shapes)
# 
#     elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
#         if device == "cuda":
#             raise RuntimeError("`tf2onnx` does not support export on CUDA device.")
#         if input_shapes is not None:
#             logger.info("`input_shapes` argument is not supported by the Tensorflow ONNX export and will be ignored.")
#         return export_tensorflow(model, config, opset, output)
# 
#     else:
#         raise RuntimeError(
#             "You either provided a PyTorch model with only TensorFlow installed, or a TensorFlow model with only PyTorch installed."
#         )
