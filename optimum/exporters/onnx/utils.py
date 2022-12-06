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

from typing import TYPE_CHECKING, Dict, Tuple, Union

import packaging
from transformers.utils import is_tf_available, is_torch_available


if TYPE_CHECKING:
    from .base import OnnxConfig

    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel

MIN_TORCH_VERSION = packaging.version.parse("1.11.0")
TORCH_VERSION = None
if is_torch_available():
    import torch

    TORCH_VERSION = packaging.version.parse(torch.__version__)

_is_torch_onnx_support_available = is_torch_available() and (MIN_TORCH_VERSION.major, MIN_TORCH_VERSION.minor) <= (
    TORCH_VERSION.major,
    TORCH_VERSION.minor,
)


def is_torch_onnx_support_available():
    return _is_torch_onnx_support_available


# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = packaging.version.parse("1.4.0")


def check_onnxruntime_requirements(minimum_version: packaging.version.Version):
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

    ort_version = packaging.version.parse(onnxruntime.__version__)
    if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
        raise ImportError(
            f"We found an older version of ONNX Runtime ({onnxruntime.__version__}) "
            f"but we require the version to be >= {minimum_version} to enable all the conversions options.\n"
            "Please update ONNX Runtime by running `pip install --upgrade onnxruntime`"
        )


def get_encoder_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "OnnxConfig"
) -> Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel"], "OnnxConfig"]]:
    """
    Returns the encoder and decoder parts of the model and their subsequent onnx configs.

    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]: A Dict containing the model and
        onnx configs for the encoder and decoder parts of the model.
    """
    models_for_export = dict()

    encoder_model = model.get_encoder()
    encoder_onnx_config = config.get_encoder_onnx_config(encoder_model.config)
    models_for_export["encoder"] = (encoder_model, encoder_onnx_config)

    decoder_model = model.get_decoder()
    decoder_onnx_config = config.get_decoder_onnx_config(decoder_model.config, config.task, use_past=False)
    models_for_export["decoder"] = (model, decoder_onnx_config)

    if config.use_past:
        decoder_onnx_config_with_past = config.get_decoder_onnx_config(
            decoder_model.config, config.task, use_past=True
        )
        models_for_export["decoder_with_past"] = (model, decoder_onnx_config_with_past)

    return models_for_export
