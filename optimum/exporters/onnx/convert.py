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

import copy
import gc
import multiprocessing as mp
import os
import traceback
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from transformers.generation import GenerationMixin
from transformers.modeling_utils import get_parameter_dtype
from transformers.utils import is_tf_available, is_torch_available

from ...onnx.utils import _get_onnx_external_constants, _get_onnx_external_data_tensors, check_model_uses_external_data
from ...utils import (
    DEFAULT_DUMMY_SHAPES,
    ONNX_WEIGHTS_NAME,
    TORCH_MINIMUM_VERSION,
    is_diffusers_available,
    is_torch_onnx_support_available,
    is_transformers_version,
    logging,
    require_numpy_strictly_lower,
)
from ...utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, MinimumVersionError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from ..utils import check_dummy_inputs_are_allowed
from .base import OnnxConfig
from .constants import UNPICKABLE_ARCHS
from .model_configs import SpeechT5OnnxConfig
from .utils import (
    MODEL_TYPES_REQUIRING_POSITION_IDS,
    PickableInferenceSession,
    _get_submodels_and_onnx_configs,
    recursive_to_device,
)


# TODO : moved back onnx imports applied in https://github.com/huggingface/optimum/pull/2114/files after refactorization

if is_torch_available():
    import torch
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel

if is_diffusers_available():
    from diffusers import DiffusionPipeline, ModelMixin

if is_tf_available():
    from transformers.modeling_tf_utils import TFPreTrainedModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DynamicAxisNameError(ValueError):
    pass


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
    use_subprocess: Optional[bool] = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
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
        use_subprocess (`Optional[bool]`, defaults to `True`):
            Launch validation of each exported model in a subprocess.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export and validation.
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

    if use_subprocess:
        logger.info("Validating models in subprocesses...")
    exceptions = []  # run all validations before raising
    for i, model_name in enumerate(models_and_onnx_configs.keys()):
        submodel, sub_onnx_config = models_and_onnx_configs[model_name]
        onnx_model_path = (
            output_dir.joinpath(onnx_files_subpaths[i])
            if onnx_files_subpaths is not None
            else output_dir.joinpath(model_name + ".onnx")
        )
        try:
            # Model validation is done in subprocesses, as ONNX Runtime has the bad habit of
            # not releasing memory once an InferenceSession is initialized.
            # Reference: https://github.com/huggingface/optimum/pull/1115
            validate_model_outputs(
                config=sub_onnx_config,
                reference_model=submodel,
                onnx_model=onnx_model_path,
                onnx_named_outputs=onnx_named_outputs[i],
                atol=atol,
                input_shapes=input_shapes,
                device=device,
                use_subprocess=use_subprocess,
                model_kwargs=model_kwargs,
            )
        except Exception as e:
            exceptions.append((onnx_model_path, e))

    if len(exceptions) != 0:
        for i, exception in enumerate(exceptions[:-1]):
            logger.error(f"Validation for the model {exception[0].as_posix()} raised: {exception[1]}")
        raise exceptions[-1][1]


def validate_model_outputs(
    config: OnnxConfig,
    reference_model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    onnx_model: Path,
    onnx_named_outputs: List[str],
    atol: Optional[float] = None,
    input_shapes: Optional[Dict] = None,
    device: str = "cpu",
    use_subprocess: Optional[bool] = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
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
        use_subprocess (`Optional[bool]`, defaults to `True`):
            Launch validation of each exported model in a subprocess.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export and validation.
    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    if use_subprocess:
        # InferenceSession do not support the fork start method with some EP: https://github.com/microsoft/onnxruntime/issues/7846
        mp.set_start_method("spawn", force=True)

        io_process = ValidationProcess(
            config, reference_model, onnx_model, onnx_named_outputs, atol, input_shapes, device, model_kwargs
        )
        io_process.start()
        io_process.join()

        if io_process.exception:
            error, traceback = io_process.exception
            raise error
    else:
        _run_validation(
            config,
            reference_model,
            onnx_model,
            onnx_named_outputs,
            atol,
            input_shapes,
            device,
            model_kwargs=model_kwargs,
        )


def _run_validation(
    config: OnnxConfig,
    reference_model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    onnx_model: Path,
    onnx_named_outputs: List[str],
    atol: Optional[float] = None,
    input_shapes: Optional[Dict] = None,
    device: str = "cpu",
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    from onnxruntime import GraphOptimizationLevel, SessionOptions

    model_kwargs = model_kwargs if model_kwargs is not None else {}

    logger.info(f"\nValidating ONNX model {onnx_model.as_posix()}...")

    if atol is None:
        atol = config.ATOL_FOR_VALIDATION

    if "diffusers" in str(reference_model.__class__) and not is_diffusers_available():
        raise ImportError("The pip package `diffusers` is required to validate diffusion ONNX models.")

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

    session = PickableInferenceSession(onnx_model.as_posix(), sess_options=session_options, providers=[provider])

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

    # Some models may modify in place the inputs, hence the copy.
    copy_reference_model_inputs = copy.deepcopy(reference_model_inputs)
    copy_reference_model_inputs = config.rename_ambiguous_inputs(copy_reference_model_inputs)

    with config.patch_model_for_export(reference_model, model_kwargs=model_kwargs):
        if is_torch_available() and isinstance(reference_model, nn.Module):
            with torch.inference_mode():
                ref_outputs = reference_model(**copy_reference_model_inputs)
        else:
            ref_outputs = reference_model(**copy_reference_model_inputs)
    ref_outputs_dict = {}

    # We flatten potential collection of outputs (i.e. past_keys) to a flat structure
    for name, value in ref_outputs.items():
        # Overwriting the output name as "present" since it is the name used for the ONNX outputs
        # ("past_key_values" being taken for the ONNX inputs)
        if name == "past_key_values":
            name = "present"
        if isinstance(value, (list, tuple)):
            onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
            value = config.flatten_output_collection_property(onnx_output_name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value

    onnx_input_names = [inp.name for inp in session.get_inputs()]

    # Possibly edit the input for the onnxruntime.InferenceSession, this is for example the case for merged
    # models where the input `use_cache_branch` is added
    reference_ort_inputs = config.generate_dummy_inputs_for_validation(
        reference_model_inputs, onnx_input_names=onnx_input_names
    )

    # We flatten potential collection of inputs (i.e. past_keys)
    onnx_inputs = {}
    for name, value in reference_ort_inputs.items():
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            onnx_inputs.update({tensor_name: pt_tensor.cpu().numpy() for tensor_name, pt_tensor in value.items()})
        elif isinstance(value, dict):
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
            f"ONNX model output names: {onnx_outputs_set}\n"
            f"Difference: {onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        onnx_output_names = ", ".join(onnx_outputs_set)
        logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_output_names})")

    if "diffusers" in str(reference_model.__class__) and not is_diffusers_available():
        raise ImportError("The pip package `diffusers` is required to validate diffusion ONNX models.")

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
        atol_msg = f"The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance {atol}:\n{msg}"

        if isinstance(config, SpeechT5OnnxConfig):
            atol_msg += "\nIMPORTANT NOTE: SpeechT5 uses a dropout at inference and the output validation of ONNX Runtime inference vs PyTorch is expected to fail. Reference: https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L727"
        raise AtolError(atol_msg)


class ValidationProcess(mp.Process):
    def __init__(
        self,
        config: OnnxConfig,
        reference_model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
        onnx_model: Path,
        onnx_named_outputs: List[str],
        atol: Optional[float] = None,
        input_shapes: Optional[Dict] = None,
        device: str = "cpu",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self.config = config
        self.reference_model = reference_model
        self.onnx_model = onnx_model
        self.onnx_named_outputs = onnx_named_outputs
        self.atol = atol
        self.input_shapes = input_shapes
        self.device = device
        self.model_kwargs = model_kwargs

    def run(self):
        try:
            _run_validation(
                config=self.config,
                reference_model=self.reference_model,
                onnx_model=self.onnx_model,
                onnx_named_outputs=self.onnx_named_outputs,
                atol=self.atol,
                input_shapes=self.input_shapes,
                device=self.device,
                model_kwargs=self.model_kwargs,
            )
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            return

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    no_dynamic_axes: bool = False,
    do_constant_folding: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
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
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
        no_dynamic_axes (bool, defaults to `False`):
            If True, disables the use of dynamic axes during ONNX export.
        do_constant_folding (bool, defaults to `True`):
            PyTorch-specific argument. If `True`, the PyTorch ONNX export will fold constants into adjacent nodes, if possible.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_onnx_config` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named outputs from
        the ONNX configuration.
    """
    from torch.onnx import export as onnx_export
    from torch.utils._pytree import tree_map

    logger.info(f"Using framework PyTorch: {torch.__version__}")
    FORCE_ONNX_EXTERNAL_DATA = os.getenv("FORCE_ONNX_EXTERNAL_DATA", "0") == "1"

    model_kwargs = model_kwargs or {}
    # num_logits_to_keep was added in transformers 4.45 and isn't added as inputs when exporting the model
    if is_transformers_version(">=", "4.45"):
        logits_to_keep_name = "logits_to_keep" if is_transformers_version(">=", "4.49") else "num_logits_to_keep"
        if logits_to_keep_name in signature(model.forward).parameters.keys():
            model_kwargs[logits_to_keep_name] = 0

    with torch.no_grad():
        model.config.return_dict = True
        model = model.eval()

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

            return value

        if device.type == "cuda" and torch.cuda.is_available():
            model.to(device)
            dummy_inputs = tree_map(remap, dummy_inputs)

        dummy_inputs = config.rename_ambiguous_inputs(dummy_inputs)

        with config.patch_model_for_export(model, model_kwargs=model_kwargs):
            check_dummy_inputs_are_allowed(model, dummy_inputs)

            inputs = config.ordered_inputs(model)
            input_names = list(inputs.keys())
            output_names = list(config.outputs.keys())

            if no_dynamic_axes:
                dynamix_axes = None
            else:
                dynamix_axes = dict(chain(inputs.items(), config.outputs.items()))

            # Export can work with named args but the dict containing named args has to be the last element of the args
            # tuple.
            onnx_export(
                model,
                (dummy_inputs,),
                f=output.as_posix(),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamix_axes,
                do_constant_folding=do_constant_folding,
                opset_version=opset,
            )

        # check if external data was exported
        onnx_model = onnx.load(str(output), load_external_data=False)
        model_uses_external_data = check_model_uses_external_data(onnx_model)

        if model_uses_external_data or FORCE_ONNX_EXTERNAL_DATA:
            tensors_paths = _get_onnx_external_data_tensors(onnx_model)
            constant_paths = _get_onnx_external_constants(onnx_model)
            logger.info("Saving external data to one file...")

            # try free model memory
            del model
            del onnx_model
            gc.collect()

            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # this will probably be too memory heavy for large models
            onnx_model = onnx.load(str(output), load_external_data=True)
            onnx.save(
                onnx_model,
                str(output),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=output.name + "_data",
                size_threshold=1024 if not FORCE_ONNX_EXTERNAL_DATA else 100,
                convert_attribute=True,
            )

            # delete previous external data
            for tensor in tensors_paths:
                os.remove(output.parent / tensor)

            for tensor in constant_paths:
                if os.path.isfile(output.parent / tensor):
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
        device (`Optional[str]`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named outputs from
        the ONNX configuration.
    """
    # This is needed to import onnx and tf2onnx because onnx is also the name of the current directory.
    import sys

    import onnx
    import tensorflow as tf
    import tf2onnx

    sys_path_backup = sys.path
    sys.path.pop(0)

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
    onnx.save(
        onnx_model,
        output.as_posix(),
        convert_attribute=True,
    )

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
    no_dynamic_axes: bool = False,
    do_constant_folding: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
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
        no_dynamic_axes (bool, defaults to `False`):
            If True, disables the use of dynamic axes during ONNX export.
        do_constant_folding (bool, defaults to `True`):
            PyTorch-specific argument. If `True`, the PyTorch ONNX export will fold constants into adjacent nodes, if possible.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_onnx_config` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).
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

        logger.info(
            f"\n***** Exporting submodel {i + 1}/{len(models_and_onnx_configs)}: {submodel.__class__.__name__} *****"
        )
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
                no_dynamic_axes=no_dynamic_axes,
                do_constant_folding=do_constant_folding,
                model_kwargs=model_kwargs,
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
    no_dynamic_axes: bool = False,
    do_constant_folding: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
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
        device (`Optional[str]`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
        disable_dynamic_axes_fix (`Optional[bool]`, defaults to `False`):
            Whether to disable the default dynamic axes fixing.
        dtype (`Optional[str]`, defaults to `None`):
            Data type to remap the model inputs to. PyTorch-only. Only `fp16` is supported.
        no_dynamic_axes (bool, defaults to `False`):
            If True, disables the use of dynamic axes during ONNX export.
        do_constant_folding (bool, defaults to `True`):
            PyTorch-specific argument. If `True`, the PyTorch ONNX export will fold constants into adjacent nodes, if possible.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_onnx_config` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).

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
        raise ImportError("The pip package `diffusers` is required to export diffusion models to ONNX.")

    if not config.is_transformers_support_available:
        import transformers

        raise MinimumVersionError(
            f"The current version of Transformers does not allow for the export of the model. Minimum required is "
            f"{config.MIN_TRANSFORMERS_VERSION}, got: {transformers.__version__}"
        )

    if is_torch_available() and isinstance(model, nn.Module):
        from ...utils.import_utils import _torch_version

        if not is_torch_onnx_support_available():
            raise MinimumVersionError(
                f"Unsupported PyTorch version, minimum required is {TORCH_MINIMUM_VERSION}, got: {_torch_version}"
            )

        if not config.is_torch_support_available:
            raise MinimumVersionError(
                f"Unsupported PyTorch version for this model. Minimum required is {config.MIN_TORCH_VERSION}, got: {_torch_version}"
            )

        export_output = export_pytorch(
            model,
            config,
            opset,
            output,
            device=device,
            input_shapes=input_shapes,
            no_dynamic_axes=no_dynamic_axes,
            do_constant_folding=do_constant_folding,
            model_kwargs=model_kwargs,
        )

    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        if model_kwargs is not None:
            raise NotImplementedError(
                "The argument `model_kwargs` is used only for PyTorch ONNX export, and unavailable for the Tensorflow export."
            )
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


def onnx_export_from_model(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline"],
    output: Union[str, Path],
    opset: Optional[int] = None,
    optimize: Optional[str] = None,
    monolith: bool = False,
    no_post_process: bool = False,
    atol: Optional[float] = None,
    do_validation: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_onnx_configs: Optional[Dict[str, "OnnxConfig"]] = None,
    fn_get_submodels: Optional[Callable] = None,
    _variant: str = "default",
    legacy: bool = False,
    preprocessors: List = None,
    device: str = "cpu",
    no_dynamic_axes: bool = False,
    task: Optional[str] = None,
    use_subprocess: bool = False,
    do_constant_folding: bool = True,
    **kwargs_shapes,
):
    """
    Full-suite ONNX export function, exporting **from a pre-loaded PyTorch or Tensorflow model**. This function is especially useful in case one needs to do modifications on the model, as overriding a forward call, before exporting to ONNX.

    Args:
        > Required parameters

        model (`Union["PreTrainedModel", "TFPreTrainedModel"]`):
            PyTorch or TensorFlow model to export to ONNX.
        output (`Union[str, Path]`):
            Path indicating the directory where to store the generated ONNX model.

        > Optional parameters

        task (`Optional[str]`, defaults to `None`):
            The task to export the model for. If not specified, the task will be auto-inferred based on the model.
        opset (`Optional[int]`, defaults to `None`):
            If specified, ONNX opset version to export the model with. Otherwise, the default opset for the given model architecture
            will be used.
        device (`str`, defaults to `"cpu"`):
            The device to use to do the export. Defaults to "cpu".
        optimize (`Optional[str]`, defaults to `None`):
            Allows to run ONNX Runtime optimizations directly during the export. Some of these optimizations are specific to
            ONNX Runtime, and the resulting ONNX will not be usable with other runtime as OpenVINO or TensorRT.
            Available options: `"O1", "O2", "O3", "O4"`. Reference: [`~optimum.onnxruntime.AutoOptimizationConfig`]
        monolith (`bool`, defaults to `False`):
            Forces to export the model as a single ONNX file.
        no_post_process (`bool`, defaults to `False`):
            Allows to disable any post-processing done by default on the exported ONNX models.
        atol (`Optional[float]`, defaults to `None`):
            If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_onnx_configs` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).
        custom_onnx_configs (`Optional[Dict[str, OnnxConfig]]`, defaults to `None`):
            Experimental usage: override the default ONNX config used for the given model. This argument may be useful for advanced users that desire a finer-grained control on the export. An example is available [here](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model).
        fn_get_submodels (`Optional[Callable]`, defaults to `None`):
            Experimental usage: Override the default submodels that are used at the export. This is
            especially useful when exporting a custom architecture that needs to split the ONNX (e.g. encoder-decoder). If unspecified with custom models, optimum will try to use the default submodels used for the given task, with no guarantee of success.
        use_subprocess (`bool`, defaults to `False`):
            Do the ONNX exported model validation in subprocesses. This is especially useful when
            exporting on CUDA device, where ORT does not release memory at inference session
            destruction. When set to `True`, the `main_export` call should be guarded in
            `if __name__ == "__main__":` block.
        _variant (`str`, defaults to `default`):
            Specify the variant of the ONNX export to use.
        legacy (`bool`, defaults to `False`):
            Disable the use of position_ids for text-generation models that require it for batched generation. Also enable to export decoder only models in three files (without + with past and the merged model). This argument is introduced for backward compatibility and will be removed in a future release of Optimum.
        no_dynamic_axes (bool, defaults to `False`):
            If True, disables the use of dynamic axes during ONNX export.
        do_constant_folding (bool, defaults to `True`):
            PyTorch-specific argument. If `True`, the PyTorch ONNX export will fold constants into adjacent nodes, if possible.
        **kwargs_shapes (`Dict`):
            Shapes to use during inference. This argument allows to override the default shapes used during the ONNX export.

    Example usage:
    ```python
    >>> from transformers import AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> # At this point, we could override some submodules, forward methods, weights, etc. from the model.

    >>> onnx_export_from_model(model, output="gpt2_onnx/")
    ```
    """
    TasksManager.standardize_model_attributes(model)

    if hasattr(model.config, "export_model_type"):
        model_type = model.config.export_model_type.replace("_", "-")
    else:
        model_type = model.config.model_type.replace("_", "-")

    library_name = TasksManager.infer_library_from_model(model)

    custom_architecture = library_name == "transformers" and model_type not in TasksManager._SUPPORTED_MODEL_TYPE

    if task is not None:
        task = TasksManager.map_from_synonym(task)
    else:
        try:
            task = TasksManager._infer_task_from_model_or_model_class(model=model)
        except (ValueError, KeyError) as e:
            raise RuntimeError(
                f"The model task could not be automatically inferred in `onnx_export_from_model`. Please provide the argument `task` with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

        if (
            not custom_architecture
            and library_name != "diffusers"
            and task + "-with-past"
            in TasksManager.get_supported_tasks_for_model_type(model_type, "onnx", library_name=library_name)
            and not monolith
        ):
            # -with-past is the default.
            task = task + "-with-past"

        logger.info(f"Automatic task detection to: {task}.")

    framework = "pt" if is_torch_available() and isinstance(model, torch.nn.Module) else "tf"

    dtype = get_parameter_dtype(model) if framework == "pt" else model.dtype

    if "bfloat16" in str(dtype):
        float_dtype = "bf16"
    elif "float16" in str(dtype):
        float_dtype = "fp16"
    else:
        float_dtype = "fp32"

    # TODO: support onnx_config.py in the model repo
    if custom_architecture and custom_onnx_configs is None:
        raise ValueError(
            f"Trying to export a {model_type} model, that is a custom or unsupported architecture, but no custom onnx configuration was passed as `custom_onnx_configs`. Please refer to https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models for an example on how to export custom models. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the model type {model_type} to be supported natively in the ONNX export."
        )

    if task.startswith("text-generation") and model.config.is_encoder_decoder:
        raise ValueError(
            f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
            f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
            f" referring to `optimum.exporters.tasks.TaskManager`'s `_TRANSFORMERS_TASKS_TO_MODEL_LOADERS`."
        )

    if legacy and model_type in MODEL_TYPES_REQUIRING_POSITION_IDS and task.startswith("text-generation"):
        logger.warning(
            f"legacy=True was specified in the ONNX export, although the model {model_type} requires position_ids for batched inference. Passing `legacy=True` is strongly discouraged, and this option will be removed in a future release. Reference: https://github.com/huggingface/optimum/pull/1381"
        )

    if library_name != "diffusers" and model_type in TasksManager._UNSUPPORTED_CLI_MODEL_TYPE:
        raise ValueError(
            f"{model_type} is not supported yet. Only {list(TasksManager._SUPPORTED_CLI_MODEL_TYPE.keys())} are supported. "
            f"If you want to support {model_type} please propose a PR or open up an issue."
        )

    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    # For MODEL_TO_PATCH_FOR_PAST architectures, when exporting the model with an input of sequence length of 1, a tracer that does not handle
    # controlflows will trace incorrectly the mask generation, resulting in incorrect attention masks for other sequence lengthss.
    # Reference: https://github.com/huggingface/transformers/blob/af3de8d87c717c4bb090f037d0d89413c195a42f/src/transformers/modeling_attn_mask_utils.py#L94
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = (
            kwargs_shapes[input_name] if input_name in kwargs_shapes else DEFAULT_DUMMY_SHAPES[input_name]
        )

        # TODO: this may be moved rather to the OnnxConfig to avoid bloating this script.
        if (
            model_type in MODEL_TO_PATCH_FOR_PAST
            and input_name == "sequence_length"
            and kwargs_shapes.get(input_name) == 1
        ):
            raise ValueError(
                f"Exporting with a sequence length of 1 a {model_type} model is not supported and can yield unexpected results."
            )

    onnx_config, models_and_onnx_configs = _get_submodels_and_onnx_configs(
        model=model,
        task=task,
        monolith=monolith,
        custom_onnx_configs=custom_onnx_configs if custom_onnx_configs is not None else {},
        custom_architecture=custom_architecture,
        float_dtype=float_dtype,
        fn_get_submodels=fn_get_submodels,
        preprocessors=preprocessors,
        _variant=_variant,
        legacy=legacy,
        library_name=library_name,
        model_kwargs=model_kwargs,
    )

    if library_name != "diffusers":
        # Ensure the requested opset is sufficient
        if opset is None:
            opset = onnx_config.DEFAULT_ONNX_OPSET
        elif opset < onnx_config.DEFAULT_ONNX_OPSET:
            logger.warning(
                f"Opset {opset} is lower than the recommended minmum opset ({onnx_config.DEFAULT_ONNX_OPSET}) to export {model_type}. "
                f"The ONNX export may fail or the exported model may be suboptimal."
            )
        if atol is None:
            atol = onnx_config.ATOL_FOR_VALIDATION
            if isinstance(atol, dict):
                atol = atol[task.replace("-with-past", "")]

        if is_transformers_version(">=", "4.44.99"):
            misplaced_generation_parameters = model.config._get_non_default_generation_parameters()
            if (
                isinstance(model, GenerationMixin)
                and model.can_generate()
                and len(misplaced_generation_parameters) > 0
            ):
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(model.generation_config, param_name, param_value)
                    setattr(model.config, param_name, None)

        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(output)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            # since v4.41.0 an exceptions will be raised when saving a generation config considered invalid
            # https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/generation/configuration_utils.py#L697
            try:
                generation_config.save_pretrained(output)
            except Exception as exception:
                logger.warning(f"The generation config is invalid and will not be saved : {exception}")

        model_name_or_path = model.config._name_or_path
        maybe_save_preprocessors(model_name_or_path, output)

        onnx_files_subpaths = [key + ".onnx" for key in models_and_onnx_configs.keys()]
    else:
        # save the subcomponent configuration
        for model_name in models_and_onnx_configs:
            subcomponent = models_and_onnx_configs[model_name][0]
            if hasattr(subcomponent, "save_config"):
                subcomponent.save_config(output / model_name)
            elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
                subcomponent.config.save_pretrained(output / model_name)

        onnx_files_subpaths = [os.path.join(name_dir, ONNX_WEIGHTS_NAME) for name_dir in models_and_onnx_configs]

        # Saving the additional components needed to perform inference.
        model.scheduler.save_pretrained(output.joinpath("scheduler"))

        feature_extractor = getattr(model, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output.joinpath("feature_extractor"))

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output.joinpath("tokenizer"))

        tokenizer_2 = getattr(model, "tokenizer_2", None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output.joinpath("tokenizer_2"))

        tokenizer_3 = getattr(model, "tokenizer_3", None)
        if tokenizer_3 is not None:
            tokenizer_3.save_pretrained(output.joinpath("tokenizer_3"))

        model.save_config(output)

    if float_dtype == "bf16":
        logger.warning(
            f"Exporting the model {model.__class__.__name__} in bfloat16 float dtype. After the export, ONNX Runtime InferenceSession with CPU/CUDA execution provider likely does not implement all operators for the bfloat16 data type, and the loading is likely to fail."
        )

    _, onnx_outputs = export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        opset=opset,
        output_dir=output,
        output_names=onnx_files_subpaths,
        input_shapes=input_shapes,
        device=device,
        dtype=float_dtype,
        no_dynamic_axes=no_dynamic_axes,
        do_constant_folding=do_constant_folding,
        model_kwargs=model_kwargs,
    )

    if optimize is not None:
        from ...onnxruntime import AutoOptimizationConfig, ORTOptimizer

        optimizer = ORTOptimizer.from_pretrained(output, file_names=onnx_files_subpaths)

        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level=optimize)

        optimization_config.disable_shape_inference = True
        optimizer.optimize(save_dir=output, optimization_config=optimization_config, file_suffix="")

    # Optionally post process the obtained ONNX file(s), for example to merge the decoder / decoder with past if any
    # TODO: treating diffusion separately is quite ugly
    if not no_post_process and library_name != "diffusers":
        try:
            logger.info("Post-processing the exported models...")
            models_and_onnx_configs, onnx_files_subpaths = onnx_config.post_process_exported_models(
                output, models_and_onnx_configs, onnx_files_subpaths
            )
        except Exception as e:
            raise Exception(
                f"The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}"
            )

    if library_name == "diffusers":
        # TODO: fix Can't pickle local object 'get_stable_diffusion_models_for_export.<locals>.<lambda>'
        use_subprocess = False
    elif model_type in UNPICKABLE_ARCHS:
        # Pickling is bugged for nn.utils.weight_norm: https://github.com/pytorch/pytorch/issues/102983
        # TODO: fix "Cowardly refusing to serialize non-leaf tensor" error for wav2vec2-conformer
        use_subprocess = False

    if device == "cpu":
        # Using multiprocessing for validation is useful only on CUDA EP that leaks memory.
        use_subprocess = False

    if do_validation is True:
        try:
            validate_models_outputs(
                models_and_onnx_configs=models_and_onnx_configs,
                onnx_named_outputs=onnx_outputs,
                atol=atol,
                output_dir=output,
                onnx_files_subpaths=onnx_files_subpaths,
                input_shapes=input_shapes,
                device=device,
                use_subprocess=use_subprocess,
                model_kwargs=model_kwargs,
            )
            logger.info(f"The ONNX export succeeded and the exported model was saved at: {output.as_posix()}")
        except ShapeError as e:
            raise e
        except AtolError as e:
            logger.warning(
                f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
            )
        except OutputMatchError as e:
            logger.warning(
                f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
            )
        except Exception as e:
            raise Exception(
                f"An error occured during validation, but the model was saved nonetheless at {output.as_posix()}. Detailed error: {e}."
            )
