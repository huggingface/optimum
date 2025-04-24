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
"""Utility functions."""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from transformers.utils import is_tf_available, is_torch_available

from ...utils import DIFFUSERS_MINIMUM_VERSION, ORT_QUANTIZE_MINIMUM_VERSION, logging
from ...utils.import_utils import (
    _diffusers_version,
    is_diffusers_available,
    is_diffusers_version,
    is_transformers_version,
)
from ..utils import (
    _get_submodels_and_export_configs,
)
from ..utils import (
    get_decoder_models_for_export as _get_decoder_models_for_export,
)
from ..utils import (
    get_diffusion_models_for_export as _get_diffusion_models_for_export,
)
from ..utils import (
    get_encoder_decoder_models_for_export as _get_encoder_decoder_models_for_export,
)
from ..utils import (
    get_sam_models_for_export as _get_sam_models_for_export,
)
from ..utils import (
    get_speecht5_models_for_export as _get_speecht5_models_for_export,
)


logger = logging.get_logger()


if is_diffusers_available():
    if not is_diffusers_version(">=", DIFFUSERS_MINIMUM_VERSION.base_version):
        raise ImportError(
            f"We found an older version of diffusers {_diffusers_version} but we require diffusers to be >= {DIFFUSERS_MINIMUM_VERSION}. "
            "Please update diffusers by running `pip install --upgrade diffusers`"
        )

if TYPE_CHECKING:
    from ..base import ExportConfig

    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel

    if is_diffusers_available():
        from diffusers import DiffusionPipeline, ModelMixin


MODEL_TYPES_REQUIRING_POSITION_IDS = {
    "codegen",
    "falcon",
    "gemma",
    "gpt2",
    "gpt-bigcode",
    "gpt-neo",
    "gpt-neox",
    "gptj",
    "imagegpt",
    "llama",
    "mistral",
    "phi",
    "phi3",
    "qwen2",
    "granite",
}


if is_transformers_version(">=", "4.45.99"):
    MODEL_TYPES_REQUIRING_POSITION_IDS.add("opt")


def check_onnxruntime_requirements(minimum_version: version.Version):
    """
    Checks that ONNX Runtime is installed and if version is recent enough.

    Args:
        minimum_version (`packaging.version.Version`):
            The minimum version allowed for the onnxruntime package.

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError(
            "ONNX Runtime doesn't seem to be currently installed. "
            "Please install ONNX Runtime by running `pip install onnxruntime`"
            " and relaunch the conversion."
        )

    ort_version = version.parse(onnxruntime.__version__)
    if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
        raise ImportError(
            f"We found an older version of ONNX Runtime ({onnxruntime.__version__}) "
            f"but we require the version to be >= {minimum_version} to enable all the conversions options.\n"
            "Please update ONNX Runtime by running `pip install --upgrade onnxruntime`"
        )


def recursive_to_device(value: Union[Tuple, List, "torch.Tensor"], device: str):
    if isinstance(value, tuple):
        value = list(value)
        for i, val in enumerate(value):
            value[i] = recursive_to_device(val, device)
        value = tuple(value)
    elif isinstance(value, list):
        for i, val in enumerate(value):
            value[i] = recursive_to_device(val, device)
    elif isinstance(value, torch.Tensor):
        value = value.to(device)

    return value


def recursive_to_dtype(
    value: Union[Tuple, List, "torch.Tensor"], dtype: Optional[torch.dtype], start_dtype: Optional[torch.dtype] = None
):
    if dtype is None:
        return value

    if isinstance(value, tuple):
        value = list(value)
        for i, val in enumerate(value):
            value[i] = recursive_to_dtype(val, dtype)
        value = tuple(value)
    elif isinstance(value, list):
        for i, val in enumerate(value):
            value[i] = recursive_to_dtype(val, dtype)
    elif isinstance(value, torch.Tensor):
        if start_dtype is None or (start_dtype is not None and value.dtype == start_dtype):
            value = value.to(dtype=dtype)

    return value


# Copied from https://github.com/microsoft/onnxruntime/issues/7846#issuecomment-850217402
class PickableInferenceSession:  # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, sess_options, providers):
        import onnxruntime as ort

        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers
        self.sess = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)

    def run(self, *args):
        return self.sess.run(*args)

    def get_outputs(self):
        return self.sess.get_outputs()

    def get_inputs(self):
        return self.sess.get_inputs()

    def __getstate__(self):
        return {"model_path": self.model_path}

    def __setstate__(self, values):
        import onnxruntime as ort

        self.model_path = values["model_path"]
        self.sess = ort.InferenceSession(self.model_path, sess_options=self.sess_options, providers=self.providers)


def _get_submodels_and_onnx_configs(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    task: str,
    monolith: bool,
    custom_onnx_configs: Dict,
    custom_architecture: bool,
    _variant: str,
    library_name: str,
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    fn_get_submodels: Optional[Callable] = None,
    preprocessors: Optional[List[Any]] = None,
    legacy: bool = False,
    model_kwargs: Optional[Dict] = None,
):
    return _get_submodels_and_export_configs(
        model,
        task,
        monolith,
        custom_onnx_configs,
        custom_architecture,
        _variant,
        library_name,
        int_dtype,
        float_dtype,
        fn_get_submodels,
        preprocessors,
        legacy,
        model_kwargs,
        exporter="onnx",
    )


DEPRECATION_WARNING_GET_MODEL_FOR_EXPORT = "The usage of `optimum.exporters.onnx.utils.get_{model_type}_models_for_export` is deprecated and will be removed in a future release, please use `optimum.exporters.utils.get_{model_type}_models_for_export` instead."


def get_diffusion_models_for_export(
    pipeline: "DiffusionPipeline",
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "ExportConfig"]]:
    logger.warning(DEPRECATION_WARNING_GET_MODEL_FOR_EXPORT.format(model_type="diffusion"))
    return _get_diffusion_models_for_export(pipeline, int_dtype, float_dtype, exporter="onnx")


def get_sam_models_for_export(model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "ExportConfig"):
    logger.warning(DEPRECATION_WARNING_GET_MODEL_FOR_EXPORT.format(model_type="sam"))
    return _get_sam_models_for_export(model, config)


def get_speecht5_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "ExportConfig", model_kwargs: Optional[Dict]
):
    logger.warning(DEPRECATION_WARNING_GET_MODEL_FOR_EXPORT.format(model_type="speecht5"))
    return _get_speecht5_models_for_export(model, config)


def get_encoder_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "ExportConfig"
) -> Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel"], "ExportConfig"]]:
    logger.warning(DEPRECATION_WARNING_GET_MODEL_FOR_EXPORT.format(model_type="encoder_decoder"))
    return _get_encoder_decoder_models_for_export(model, config)


def get_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    config: "ExportConfig",
    legacy: bool = False,
) -> Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel"], "ExportConfig"]]:
    logger.warning(DEPRECATION_WARNING_GET_MODEL_FOR_EXPORT.format(model_type="decoder"))
    return _get_decoder_models_for_export(model, config, legacy)
