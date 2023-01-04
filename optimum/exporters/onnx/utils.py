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

import copy
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import packaging
import torch
from transformers.utils import is_tf_available, is_torch_available

from ...utils import ORT_QUANTIZE_MINIMUM_VERSION, TORCH_MINIMUM_VERSION, is_diffusers_available
from ..tasks import TasksManager


if TYPE_CHECKING:
    from .base import OnnxConfig

    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin, StableDiffusionPipeline


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
    models_for_export = {}

    encoder_model = model.get_encoder()
    encoder_onnx_config = config.with_behavior("encoder")
    models_for_export["encoder_model"] = (encoder_model, encoder_onnx_config)

    decoder_onnx_config = config.with_behavior("decoder", use_past=False)
    models_for_export["decoder_model"] = (model, decoder_onnx_config)

    if config.use_past:
        decoder_onnx_config_with_past = config.with_behavior("decoder", use_past=True)
        models_for_export["decoder_with_past_model"] = (model, decoder_onnx_config_with_past)

    return models_for_export


def get_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    config: "OnnxConfig",
) -> Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel"], "OnnxConfig"]]:
    """
    Returns two versions of the decoder that can be used together to perform fast generation:

        1. The first one takes regular inputs, and outputs the result along with past key/values.
        2. The second one takes regular inputs and past key/values, and outputs the result along with the updated past
        key/values.


    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.

    Returns:
        `Dict[str, Tuple[Union[PreTrainedModel, TFPreTrainedModel], OnnxConfig]]: A Dict containing the model and
        onnx configs for the encoder and decoder parts of the model.
    """
    models_for_export = {}

    onnx_config = config.__class__(
        model.config, task=config.task, use_past_in_inputs=False, use_present_in_outputs=True
    )
    models_for_export["decoder_model"] = (model, onnx_config)

    if config.use_past:
        onnx_config_with_past = config.__class__(model.config, task=config.task, use_past=True)
        models_for_export["decoder_with_past_model"] = (model, onnx_config_with_past)

    return models_for_export


def get_stable_diffusion_models_for_export(
    pipeline: "StableDiffusionPipeline",
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "OnnxConfig"]]:
    """
    Returns the components of a Stable Diffusion model and their subsequent onnx configs.

    Args:
        pipeline ([`StableDiffusionPipeline`]):
            The model to export.

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]: A Dict containing the model and
        onnx configs for the different components of the model.
    """
    models_for_export = dict()

    # Text encoder
    text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.text_encoder, exporter="onnx", task="default"
    )
    text_encoder_onnx_config = text_encoder_config_constructor(pipeline.text_encoder.config)
    models_for_export["text_encoder"] = (pipeline.text_encoder, text_encoder_onnx_config)

    # U-NET
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.unet, exporter="onnx", task="semantic-segmentation", model_type="unet"
    )
    unet_onnx_config = onnx_config_constructor(pipeline.unet.config)
    models_for_export["unet"] = (pipeline.unet, unet_onnx_config)

    # VAE
    vae = copy.deepcopy(pipeline.vae)
    vae.forward = lambda latent_sample: vae.decode(z=latent_sample)
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae, exporter="onnx", task="semantic-segmentation", model_type="vae"
    )
    vae_onnx_config = vae_config_constructor(vae.config)
    models_for_export["vae"] = (vae, vae_onnx_config)

    return models_for_export


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
