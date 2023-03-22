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

import os
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import onnx
from transformers.utils import is_tf_available, is_torch_available

from ...onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from ...utils import (
    TORCH_MINIMUM_VERSION,
    is_diffusers_available,
    is_torch_onnx_support_available,
    logging,
    require_numpy_strictly_lower,
)
from ..error_utils import AtolError, OutputMatchError, ShapeError
from .base import OnnxConfig
from .utils import recursive_to_device, recursive_to_dtype


if is_torch_available():
    import torch
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel
    from transformers.pytorch_utils import is_torch_less_than_1_11

if is_diffusers_available():
    from diffusers import ModelMixin

if is_tf_available():
    from transformers.modeling_tf_utils import TFPreTrainedModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DynamicAxisNameError(ValueError):
    pass


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

    forward = model.forward if is_torch_available() and isinstance(model, nn.Module) else model.call
    forward_parameters = signature(forward).parameters
    forward_inputs_set = set(forward_parameters.keys())
    dummy_input_names = set(dummy_input_names)

    # We are fine if config_inputs has more keys than model_inputs
    if not dummy_input_names.issubset(forward_inputs_set):
        raise ValueError(
            f"Config dummy inputs are not a subset of the model inputs: {dummy_input_names} vs {forward_inputs_set}"
        )


def validate_models_outputs(
    models_and_onnx_configs: Dict[
        str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]
    ],
    onnx_named_outputs: List[List[str]],
    output_dir: Path,
    atol: Optional[float] = None,
    onnx_files_subpaths: Optional[List[str]] = None,
    input_shapes: Optional[Dict] = None,
    device: str = "cpu",
    dtype: Optional["torch.dtype"] = None,
):
    """
    Validates the export of several models, by checking that the outputs from both the reference and the exported model match.
    The following method validates the ONNX models exported using the `export_models` method.

    Args:
        models_and_onnx_configs (`Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]]):
            A dictionnary containing the models to validate and their corresponding onnx configs.
        onnx_named_outputs (`List[List[str]]`):
            The names of the outputs to check.
        output_dir (`Path`):
            Output directory where the exported ONNX models are stored.
        atol (`Optional[float]`, defaults to `None`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.
        onnx_files_subpaths (`Optional[List[str]]`, defaults to `None`):
            The relative paths from `output_dir` to the ONNX files to do validation on. The order must be the same as the order of submodels
            in the ordered dict `models_and_onnx_configs`. If None, will use the keys from the `models_and_onnx_configs` as names.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes to validate the ONNX model on.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX models will be validated. Either `cpu` or `cuda`. Validation on a CUDA device is supported only for PyTorch.
        dtype (`Optional[torch.dtype]`, defaults to `None`):
            Data type of the inputs to perform validation on. Validation on float16 is supported only for PyTorch.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    if len(onnx_named_outputs) != len(models_and_onnx_configs.keys()):
        raise ValueError(
            f"Invalid number of ONNX named outputs. Required {len(models_and_onnx_configs.keys())}, Provided {len(onnx_named_outputs)}"
        )

    if onnx_files_subpaths is not None and len(onnx_files_subpaths) != len(models_and_onnx_configs):
        raise ValueError(
            f"Provided custom names {onnx_files_subpaths} for the validation of {len(models_and_onnx_configs)} models. Please provide the same number of ONNX file names as models to export."
        )

    exceptions = []  # run all validations before raising
    onnx_paths = []
    for i, model_name in enumerate(models_and_onnx_configs.keys()):
        submodel, sub_onnx_config = models_and_onnx_configs[model_name]
        onnx_model_path = (
            output_dir.joinpath(onnx_files_subpaths[i])
            if onnx_files_subpaths is not None
            else output_dir.joinpath(model_name + ".onnx")
        )
        onnx_paths.append(onnx_model_path)
        try:
            validate_model_outputs(
                config=sub_onnx_config,
                reference_model=submodel,
                onnx_model=onnx_model_path,
                onnx_named_outputs=onnx_named_outputs[i],
                atol=atol,
                input_shapes=input_shapes,
                device=device,
                dtype=dtype,
            )
        except Exception as e:
            exceptions.append(e)

    if len(exceptions) != 0:
        for i, exception in enumerate(exceptions[:-1]):
            logger.error(f"Validation {i} for the model {onnx_paths[i].as_posix()} raised: {exception}")
        raise exceptions[-1]


def validate_model_outputs(
    config: OnnxConfig,
    reference_model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    onnx_model: Path,
    onnx_named_outputs: List[str],
    atol: Optional[float] = None,
    input_shapes: Optional[Dict] = None,
    device: str = "cpu",
    dtype: Optional["torch.dtype"] = None,
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
        atol (`Optional[float]`, defaults to `None`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes to validate the ONNX model on.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be validated. Either `cpu` or `cuda`. Validation on a CUDA device is supported only for PyTorch.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

    logger.info(f"Validating ONNX model {onnx_model.as_posix()}...")

    if atol is None:
        atol = config.ATOL_FOR_VALIDATION

    if "diffusers" in str(reference_model.__class__) and not is_diffusers_available():
        raise ImportError("The pip package `diffusers` is required to validate stable diffusion ONNX models.")

    framework = "pt" if is_torch_available() and isinstance(reference_model, nn.Module) else "tf"

    if input_shapes is None:
        input_shapes = {}  # will use the defaults from DEFAULT_DUMMY_SHAPES
    reference_model_inputs = config.generate_dummy_inputs(framework=framework, **input_shapes)

    # Create ONNX Runtime session
    session_options = SessionOptions()
    # We could well set ORT_DISABLE_ALL here, but it makes CUDA export with O4 of gpt_neo fail
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC

    if device.startswith("cuda"):
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    session = InferenceSession(onnx_model.as_posix(), sess_options=session_options, providers=[provider])

    # Sometimes the exported model can have more outputs than what is specified in the ONNX config because the original
    # PyTorch model has more outputs that were forgotten in the config, so we check for that.
    all_onnx_outputs = {output.name for output in session.get_outputs()}
    config_outputs = set(config.outputs)
    if all_onnx_outputs != config_outputs:
        if len(all_onnx_outputs) > len(config_outputs):
            diff = all_onnx_outputs - config_outputs
        else:
            diff = config_outputs - all_onnx_outputs

        raise OutputMatchError(
            "The exported ONNX model does not have the exact same outputs as what is provided in "
            f"{config.__class__.__name__}. Difference: {', '.join(diff)}"
        )

    # Sometimes the exported model can have axes that are inferred as dynamic axes but were not specified as such in
    # the ONNX Config: it was either an error on the config side, or an error on the ONNX side inferring a dynamic axis
    # that is actually static.
    # The `OnnxConfig.fix_dynamic_axes` method should fix that at export time, but it is still worth checking here.
    all_config_dynamic_axes_names = set()
    for input_ in config.inputs.values():
        all_config_dynamic_axes_names |= set(input_.values())
    for output in config.outputs.values():
        all_config_dynamic_axes_names |= set(output.values())

    for node in session.get_outputs():
        for idx, axis in enumerate(node.shape):
            if isinstance(axis, str) and axis not in all_config_dynamic_axes_names:
                raise DynamicAxisNameError(
                    f"The axis {idx} of input / output node called {node.name} has an unknown name: {axis}"
                )

    # Compute outputs from the reference model
    if is_torch_available() and isinstance(reference_model, nn.Module):
        reference_model.to(device)

        for key, value in reference_model_inputs.items():
            reference_model_inputs[key] = recursive_to_device(value=value, device=device)
            reference_model_inputs[key] = recursive_to_dtype(
                value=reference_model_inputs[key], dtype=dtype, start_dtype=torch.float32
            )

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

    # Possibly edit the input for the onnxruntime.InferenceSession, this is for example the case for merged
    # models where the input `use_cache_branch` is added
    reference_ort_inputs = config.generate_dummy_inputs_for_validation(reference_model_inputs)

    # We flatten potential collection of inputs (i.e. past_keys)
    onnx_inputs = {}
    for name, value in reference_ort_inputs.items():
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            onnx_inputs.update({tensor_name: pt_tensor.cpu().numpy() for tensor_name, pt_tensor in value.items()})
        else:
            onnx_inputs[name] = value.cpu().numpy()

    # Compute outputs from the ONNX model
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

    # Modify the ONNX output names to match the reference model output names
    onnx_to_torch = {v: k for k, v in config.torch_to_onnx_output_map.items()}
    onnx_named_outputs = [onnx_to_torch.get(k, k) for k in onnx_named_outputs]

    # Check we have a subset of the keys into onnx_outputs against ref_outputs
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        raise OutputMatchError(
            "ONNX model output names do not match reference model output names.\n"
            f"Reference model output names: {ref_outputs_set}\n"
            f"ONNX model output names: {onnx_outputs_set}"
            f"Difference: {onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        onnx_output_names = ", ".join(onnx_outputs_set)
        logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_output_names})")

    if "diffusers" in str(reference_model.__class__) and not is_diffusers_available():
        raise ImportError("The pip package `diffusers` is required to validate stable diffusion ONNX models.")

    # Check the shape and values match
    shape_failures = []
    value_failures = []
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        if is_torch_available() and isinstance(reference_model, nn.Module):
            ref_value = ref_outputs_dict[name].detach().cpu().numpy()
        else:
            ref_value = ref_outputs_dict[name].cpu().numpy()
        logger.info(f'\t- Validating ONNX Model output "{name}":')

        # Shape
        if not ort_value.shape == ref_value.shape:
            logger.error(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            shape_failures.append((name, ref_value.shape, ort_value.shape))
        else:
            logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

        # Values
        try:
            if not np.allclose(ref_value, ort_value, atol=atol):
                max_diff = np.amax(np.abs(ref_value - ort_value))
                logger.error(f"\t\t-[x] values not close enough, max diff: {max_diff} (atol: {atol})")
                value_failures.append((name, max_diff))
            else:
                logger.info(f"\t\t-[✓] all values close (atol: {atol})")
        except Exception:
            # If shapes do not match, it is possible that the np.allclose call fails, since we raise the proper issue
            # right after, we do not do anything here.
            pass

    if shape_failures:
        msg = "\n".join(f"- {t[0]}: got {t[1]} (reference) and {t[2]} (ONNX)" for t in shape_failures)
        raise ShapeError(f"Output shapes do not match between reference model and ONNX exported model:\n{msg}")

    if value_failures:
        msg = "\n".join(f"- {t[0]}: max diff = {t[1]}" for t in value_failures)
        raise AtolError(
            f"The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance {atol}:\n{msg}"
        )


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
    dtype: Optional["torch.dtype"] = None,
    input_shapes: Optional[Dict] = None,
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
            Path to save the exported ONNX file to.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        dtype (`Optional[torch.dtype]`, defaults to `None`):
            Data type to remap the model inputs to. PyTorch-only. Only `torch.float16` is supported.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named outputs from
        the ONNX configuration.
    """
    from torch.onnx import export as onnx_export
    from torch.utils._pytree import tree_map

    logger.info(f"Using framework PyTorch: {torch.__version__}")
    FORCE_ONNX_EXTERNAL_DATA = os.getenv("FORCE_ONNX_EXTERNAL_DATA", "0") == "1"

    with torch.no_grad():
        model.config.return_dict = True
        model.eval()

        # Check if we need to override certain configuration item
        if config.values_override is not None:
            logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
            for override_config_key, override_config_value in config.values_override.items():
                logger.info(f"\t- {override_config_key} -> {override_config_value}")
                setattr(model.config, override_config_key, override_config_value)

        if input_shapes is None:
            input_shapes = {}  # will use the defaults from DEFAULT_DUMMY_SHAPES

        # Check that inputs match, and order them properly
        dummy_inputs = config.generate_dummy_inputs(framework="pt", **input_shapes)

        device = torch.device(device)

        def remap(value):
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                value = value.to(dtype=dtype)

            return value

        if device.type == "cuda" and torch.cuda.is_available():
            model.to(device)
            dummy_inputs = tree_map(remap, dummy_inputs)
        check_dummy_inputs_are_allowed(model, dummy_inputs)
        inputs = config.ordered_inputs(model)
        input_names = list(inputs.keys())
        output_names = list(config.outputs.keys())

        # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
        # so we check the torch version for backwards compatibility
        if is_torch_less_than_1_11:
            raise RuntimeError("The ONNX export using the PyTorch framework is only supported for v1.11+")
        else:
            with config.patch_model_for_export(model):
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

            # check if external data was exported
            # TODO: this is quite inefficient as we load in memory if models are <2GB without external data
            onnx_model = onnx.load(str(output), load_external_data=False)
            model_uses_external_data = check_model_uses_external_data(onnx_model)

            if model_uses_external_data or FORCE_ONNX_EXTERNAL_DATA:
                tensors_paths = _get_onnx_external_data_tensors(onnx_model)
                logger.info("Saving external data to one file...")

                # try free model memory
                del model
                del onnx_model

                onnx_model = onnx.load(
                    str(output), load_external_data=True
                )  # this will probably be too memory heavy for large models
                onnx.save(
                    onnx_model,
                    str(output),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=output.name + "_data",
                    size_threshold=1024 if not FORCE_ONNX_EXTERNAL_DATA else 0,
                )

                # delete previous external data
                for tensor in tensors_paths:
                    os.remove(output.parent / tensor)

    return input_names, output_names


@require_numpy_strictly_lower("1.24.0", "The Tensorflow ONNX export only supports numpy<1.24.0.")
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
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named outputs from
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

    input_signature = []
    for key, tensor in dummy_inputs.items():
        shape = [tensor.shape[i] for i in range(tensor.ndim)]
        for idx, _ in config.inputs[key].items():
            shape[idx] = None

        input_signature.append(tf.TensorSpec(shape, dtype=tensor.dtype, name=key))

    with config.patch_model_for_export(model):
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=opset)
    onnx.save(onnx_model, output.as_posix())

    return input_names, output_names


def export_models(
    models_and_onnx_configs: Dict[
        str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]
    ],
    output_dir: Path,
    opset: Optional[int] = None,
    output_names: Optional[List[str]] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    disable_dynamic_axes_fix: Optional[bool] = False,
    dtype: Optional[str] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Exports a Pytorch or TensorFlow encoder decoder model to an ONNX Intermediate Representation.
    The following method exports the encoder and decoder components of the model as separate
    ONNX files.

    Args:
        models_and_onnx_configs (`Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`, `ModelMixin`], `OnnxConfig`]]):
            A dictionnary containing the models to export and their corresponding onnx configs.
        output_dir (`Path`):
            Output directory to store the exported ONNX models.
        opset (`Optional[int]`, defaults to `None`):
            The version of the ONNX operator set to use.
        output_names (`Optional[List[str]]`, defaults to `None`):
            The names to use for the exported ONNX files. The order must be the same as the order of submodels in the ordered dict `models_and_onnx_configs`.
            If None, will use the keys from `models_and_onnx_configs` as names.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
        disable_dynamic_axes_fix (`Optional[bool]`, defaults to `False`):
            Whether to disable the default dynamic axes fixing.
        dtype (`Optional[str]`, defaults to `None`):
            Data type to remap the model inputs to. PyTorch-only. Only `fp16` is supported.
    Returns:
        `Tuple[List[List[str]], List[List[str]]]`: A tuple with an ordered list of the model's inputs, and the named
        outputs from the ONNX configuration.
    """
    outputs = []

    if output_names is not None and len(output_names) != len(models_and_onnx_configs):
        raise ValueError(
            f"Provided custom names {output_names} for the export of {len(models_and_onnx_configs)} models. Please provide the same number of names as models to export."
        )

    for i, model_name in enumerate(models_and_onnx_configs.keys()):
        submodel, sub_onnx_config = models_and_onnx_configs[model_name]
        output_name = output_names[i] if output_names is not None else Path(model_name + ".onnx")

        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        outputs.append(
            export(
                model=submodel,
                config=sub_onnx_config,
                output=output_path,
                opset=opset,
                device=device,
                input_shapes=input_shapes,
                disable_dynamic_axes_fix=disable_dynamic_axes_fix,
                dtype=dtype,
            )
        )

    outputs = list(map(list, zip(*outputs)))
    return outputs


def export(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    output: Path,
    opset: Optional[int] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    disable_dynamic_axes_fix: Optional[bool] = False,
    dtype: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a Pytorch or TensorFlow model to an ONNX Intermediate Representation.

    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported ONNX model.
        opset (`Optional[int]`, defaults to `None`):
            The version of the ONNX operator set to use.
        device (`str`, *optional*, defaults to `cpu`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
        disable_dynamic_axes_fix (`Optional[bool]`, defaults to `False`):
            Whether to disable the default dynamic axes fixing.
        dtype (`Optional[str]`, defaults to `None`):
            Data type to remap the model inputs to. PyTorch-only. Only `fp16` is supported.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named outputs from
        the ONNX configuration.
    """
    if not (is_torch_available() or is_tf_available()):
        raise ImportError(
            "Cannot convert because neither PyTorch nor TensorFlow are installed. "
            "Please install torch or tensorflow first."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    export_output = None

    if opset is None:
        opset = config.DEFAULT_ONNX_OPSET

    if "diffusers" in str(model.__class__) and not is_diffusers_available():
        raise ImportError("The pip package `diffusers` is required to export stable diffusion models to ONNX.")

    if is_torch_available() and isinstance(model, nn.Module):
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

        torch_dtype = None
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype is not None:
            raise ValueError("Unsupported dtype, supported dtypes are: `torch.float16`.")

        export_output = export_pytorch(
            model, config, opset, output, device=device, input_shapes=input_shapes, dtype=torch_dtype
        )

    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        if device == "cuda":
            raise RuntimeError("`tf2onnx` does not support export on CUDA device.")
        if input_shapes is not None:
            logger.info("`input_shapes` argument is not supported by the Tensorflow ONNX export and will be ignored.")
        export_output = export_tensorflow(model, config, opset, output)

    else:
        raise RuntimeError(
            "You either provided a PyTorch model with only TensorFlow installed, or a TensorFlow model with only PyTorch installed."
        )

    if not disable_dynamic_axes_fix:
        config.fix_dynamic_axes(output, device=device, input_shapes=input_shapes, dtype=dtype)
    return export_output
