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

from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from transformers.utils import is_tf_available, is_torch_available

from ...utils import TORCH_MINIMUM_VERSION, is_torch_onnx_support_available, logging
from ..tasks import TasksManager
from .base import OnnxConfig


if is_torch_available():
    from transformers.modeling_utils import PreTrainedModel
    from transformers.pytorch_utils import is_torch_less_than_1_11

    from diffusers import ModelMixin

if is_tf_available():
    from transformers.modeling_tf_utils import TFPreTrainedModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def check_dummy_inputs_are_allowed(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], dummy_input_names: Iterable[str]
):
    """
    Checks that the dummy inputs from the ONNX config is a subset of the allowed inputs for `model`.
    Args:
        model (`Union[transformers.PreTrainedModel, transformers.TFPreTrainedModel`]):
            The model instance.
        model_inputs (`Iterable[str]`):
            The model input names.
    """
    forward = (
        model.forward
        if is_torch_available() and issubclass(type(model), (PreTrainedModel, ModelMixin))
        else model.call
    )
    forward_parameters = signature(forward).parameters
    forward_inputs_set = set(forward_parameters.keys())
    dummy_input_names = set(dummy_input_names)

    # We are fine if config_inputs has more keys than model_inputs
    if not dummy_input_names.issubset(forward_inputs_set):
        raise ValueError(
            f"Config dummy inputs are not a subset of the model inputs: {dummy_input_names} vs {forward_inputs_set}"
        )


def validate_models_outputs(
    models: Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]],
    onnx_named_outputs: List[str],
    atol: float,
    output_dir: Path,
    output_names: Optional[List[str]] = None,
):
    """
    Validates the export of several models, by checking that the outputs from both the reference and the exported model match.
    The following method validates the ONNX models exported using the `export_models` method.

    Args:
        models (`Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]]):
            A dictionnary containing the models to validate and their corresponding onnx configs.
        onnx_named_outputs (`List[str]`):
            The names of the outputs to check.
        atol (`float`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.
        output_dir (`Path`):
            Output directory where the exported ONNX models are stored.
        output_names (`Optional[List[str]]`, defaults to `None`):
            The names to use for the exported ONNX files. The order must be the same as the order of submodels in the ordered dict `models`.
            If None, will use the keys from the `models` as names.
    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """

    if len(onnx_named_outputs) != len(models.keys()):
        raise ValueError(
            f"Invalid number of ONNX named outputs. Required {len(models.keys())}, Provided {len(onnx_named_outputs)}"
        )

    if output_names is not None and len(output_names) != len(models):
        raise ValueError(
            f"Provided custom names {output_names} for the validation of {len(models)} models. Please provide the same number of ONNX file names as models to export."
        )

    for i, model_name in enumerate(models.keys()):
        submodel, sub_onnx_config = models[model_name]
        onnx_model_path = (
            output_dir.joinpath(output_names[i])
            if output_names is not None
            else output_dir.joinpath(model_name + ".onnx")
        )
        validate_model_outputs(
            sub_onnx_config,
            submodel,
            onnx_model_path,
            onnx_named_outputs[i],
            atol,
        )


def validate_model_outputs(
    config: OnnxConfig,
    reference_model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    onnx_model: Path,
    onnx_named_outputs: List[str],
    atol: float,
):
    """
    Validates the export by checking that the outputs from both the reference and the exported model match.

    Args:
        config ([`~OnnxConfig`]:
            The configuration used to export the model.
        reference_model ([`~PreTrainedModel`] or [`~TFPreTrainedModel`]):
            The model used for the export.
        onnx_model (`Path`):
            The path to the exported model.
        onnx_named_outputs (`List[str]`):
            The names of the outputs to check.
        atol (`float`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    from onnxruntime import InferenceSession, SessionOptions

    logger.info("Validating ONNX model...")

    framework = (
        "pt" if is_torch_available() and issubclass(type(reference_model), (PreTrainedModel, ModelMixin)) else "tf"
    )
    reference_model_inputs = config.generate_dummy_inputs(framework=framework)

    # Create ONNX Runtime session
    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options, providers=["CPUExecutionProvider"])

    # Compute outputs from the reference model
    if is_torch_available() and issubclass(type(reference_model), (PreTrainedModel, ModelMixin)):
        reference_model.to("cpu")

    ref_outputs = reference_model(**reference_model_inputs)
    ref_outputs_dict = {}

    # We flatten potential collection of outputs (i.e. past_keys) to a flat structure
    for name, value in ref_outputs.items():
        # Overwriting the output name as "present" since it is the name used for the ONNX outputs
        # ("past_key_values" being taken for the ONNX inputs)
        if name == "past_key_values":
            name = "present"
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value

    # Create onnxruntime inputs from the reference model inputs
    reference_model_inputs_for_validation = config.generate_dummy_inputs_for_validation(reference_model_inputs)

    # We flatten potential collection of inputs (i.e. past_keys)
    onnx_inputs = {}
    for name, value in reference_model_inputs_for_validation.items():
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            onnx_inputs.update({tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()})
        else:
            onnx_inputs[name] = value.numpy()

    # Compute outputs from the ONNX model
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

    # Modify the ONNX output names to match the reference model output names
    onnx_named_outputs = config.output_names_for_validation(onnx_named_outputs)

    # Check we have a subset of the keys into onnx_outputs against ref_outputs
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        raise ValueError(
            "ONNX model output names do not match reference model output names.\n"
            f"Reference model output names: {ref_outputs_set}\n"
            f"ONNX model output names: {onnx_outputs_set}"
            f"Difference: {onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        onnx_output_names = ", ".join(onnx_outputs_set)
        logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_output_names})")

    # Check the shape and values match
    shape_failures = []
    value_failures = []
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        if is_torch_available() and issubclass(type(reference_model), (PreTrainedModel, ModelMixin)):
            ref_value = ref_outputs_dict[name].detach().numpy()
        else:
            ref_value = ref_outputs_dict[name].numpy()
        logger.info(f'\t- Validating ONNX Model output "{name}":')

        # Shape
        if not ort_value.shape == ref_value.shape:
            logger.error(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            shape_failures.append((name, ref_value.shape, ort_value.shape))
        else:
            logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

        # Values
        if not np.allclose(ref_value, ort_value, atol=atol):
            max_diff = np.amax(np.abs(ref_value - ort_value))
            logger.error(f"\t\t-[x] values not close enough, max diff: {max_diff} (atol: {atol})")
            value_failures.append((name, max_diff))
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")

    if shape_failures:
        msg = "\n".join(f"- {t[0]}: got {t[1]} (reference) and {t[2]} (ONNX)" for t in shape_failures)
        raise ValueError(f"Output shapes do not match between reference model and ONNX exported model:\n{msg}")

    if value_failures:
        msg = "\n".join(f"- {t[0]}: max diff = {t[1]}" for t in value_failures)
        raise ValueError(f"Output values do not match between reference model and ONNX exported model:\n{msg}")


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to an ONNX Intermediate Representation.

    Args:
        model ([`PreTrainedModel`]):
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
    import torch
    from torch.onnx import export as onnx_export
    from torch.utils._pytree import tree_map

    logger.info(f"Using framework PyTorch: {torch.__version__}")

    with torch.no_grad():
        model.config.return_dict = True
        model.eval()

        # Check if we need to override certain configuration item
        if config.values_override is not None:
            logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
            for override_config_key, override_config_value in config.values_override.items():
                logger.info(f"\t- {override_config_key} -> {override_config_value}")
                setattr(model.config, override_config_key, override_config_value)

        # Check that inputs match, and order them properly
        dummy_inputs = config.generate_dummy_inputs(framework="pt")
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            model.to(device)
            dummy_inputs = tree_map(
                lambda value: value.to(device) if isinstance(value, torch.Tensor) else value, dummy_inputs
            )
        check_dummy_inputs_are_allowed(model, dummy_inputs)
        inputs = config.ordered_inputs(model)
        input_names = list(inputs.keys())
        output_names = list(config.outputs.keys())

        config.patch_ops()

        # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
        # so we check the torch version for backwards compatibility
        if is_torch_less_than_1_11:
            raise RuntimeError("The ONNX export using the PyTorch framework is only supported for v1.11+")
        else:
            # Export can work with named args but the dict containing named args has to be the last element of the args
            # tuple.
            onnx_export(
                model,
                (dummy_inputs,),
                f=output.as_posix(),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={name: axes for name, axes in chain(inputs.items(), config.outputs.items())},
                do_constant_folding=True,
                opset_version=opset,
            )

        config.restore_ops()

    return input_names, output_names


def export_tensorflow(
    model: "TFPreTrainedModel",
    config: OnnxConfig,
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
    import sys

    import tensorflow as tf

    sys_path_backup = sys.path
    sys.path.pop(0)
    import onnx
    import tf2onnx

    sys.path = sys_path_backup

    logger.info(f"Using framework TensorFlow: {tf.__version__}")

    model.config.return_dict = True

    # Check if we need to override certain configuration item
    if config.values_override is not None:
        logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in config.values_override.items():
            logger.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    # Ensure inputs match
    dummy_inputs = config.generate_dummy_inputs(framework="tf")
    check_dummy_inputs_are_allowed(model, dummy_inputs)

    inputs = config.ordered_inputs(model)
    input_names = list(inputs.keys())
    output_names = list(config.outputs.keys())

    config.patch_ops()
    input_signature = [tf.TensorSpec.from_tensor(tensor, name=key) for key, tensor in dummy_inputs.items()]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=opset)
    onnx.save(onnx_model, output.as_posix())
    config.restore_ops()

    return input_names, output_names


def export_models(
    models: Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]],
    opset: int,
    output_dir: Path,
    output_names: Optional[List[str]] = None,
    device: str = "cpu",
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Exports a Pytorch or TensorFlow encoder decoder model to an ONNX Intermediate Representation.
    The following method exports the encoder and decoder components of the model as separate
    ONNX files.

    Args:
        models (`Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]]):
             A dictionnary containing the models to export and their corresponding onnx configs.
        opset (`int`):
            The version of the ONNX operator set to use.
        output_dir (`Path`):
            Output directory to store the exported ONNX models.
        output_names (`Optional[List[str]]`, defaults to `None`):
            The names to use for the exported ONNX files. The order must be the same as the order of submodels in the ordered dict `models`.
            If None, will use the keys from `models` as names.
        device (`str`, *optional*, defaults to `cpu`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
    Returns:
        `Tuple[List[List[str]], List[List[str]]]`: A tuple with an ordered list of the model's inputs, and the named
        inputs from the ONNX configuration.
    """
    outputs = []

    if output_names is not None and len(output_names) != len(models):
        raise ValueError(
            f"Provided custom names {output_names} for the export of {len(models)} models. Please provide the same number of names as models to export."
        )

    for i, model_name in enumerate(models.keys()):
        submodel, sub_onnx_config = models[model_name]
        output_path = (
            output_dir.joinpath(output_names[i])
            if output_names is not None
            else output_dir.joinpath(model_name + ".onnx")
        )
        outputs.append(
            export(
                submodel,
                sub_onnx_config,
                opset,
                output_path,
                device=device,
            )
        )

    outputs = list(map(list, zip(*outputs)))
    return outputs


def export(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Exports a Pytorch or TensorFlow model to an ONNX Intermediate Representation.

    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
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
    if not (is_torch_available() or is_tf_available()):
        raise ImportError(
            "Cannot convert because neither PyTorch nor TensorFlow are installed. "
            "Please install torch or tensorflow first."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    if is_torch_available() and issubclass(type(model), (PreTrainedModel, ModelMixin)):
        from ...utils import torch_version

        if not is_torch_onnx_support_available():
            raise AssertionError(
                f"Unsupported PyTorch version, minimum required is {TORCH_MINIMUM_VERSION}, got: {torch_version}"
            )

        if not config.is_torch_support_available:
            logger.warning(
                f"Unsupported PyTorch version for this model. Minimum required is {config.MIN_TORCH_VERSION},"
                f" got: {torch.__version__}"
            )
        return export_pytorch(model, config, opset, output, device=device)

    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        if device == "cuda":
            raise RuntimeError("`tf2onnx` does not support export on CUDA device.")
        return export_tensorflow(model, config, opset, output)

    else:
        raise RuntimeError(
            "You either provided a PyTorch model with only TensorFlow installed, or a TensorFlow model with only PyTorch installed."
        )
