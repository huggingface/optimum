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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import packaging
import torch
from transformers.utils import is_tf_available, is_torch_available

from ...utils import (
    DIFFUSERS_MINIMUM_VERSION,
    ORT_QUANTIZE_MINIMUM_VERSION,
    check_if_diffusers_greater,
    is_diffusers_available,
    logging,
)
from ...utils.import_utils import _diffusers_version
from ..tasks import TasksManager
from .constants import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME


logger = logging.get_logger()


if is_diffusers_available():
    if not check_if_diffusers_greater(DIFFUSERS_MINIMUM_VERSION.base_version):
        raise ImportError(
            f"We found an older version of diffusers {_diffusers_version} but we require diffusers to be >= {DIFFUSERS_MINIMUM_VERSION}. "
            "Please update diffusers by running `pip install --upgrade diffusers`"
        )
    from diffusers.models.attention_processor import (
        Attention,
        AttnAddedKVProcessor,
        AttnAddedKVProcessor2_0,
        AttnProcessor,
        AttnProcessor2_0,
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
    )

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
    models_for_export[ONNX_ENCODER_NAME] = (encoder_model, encoder_onnx_config)

    decoder_onnx_config = config.with_behavior("decoder", use_past=False)
    models_for_export[ONNX_DECODER_NAME] = (model, decoder_onnx_config)

    if config.use_past:
        decoder_onnx_config_with_past = config.with_behavior("decoder", use_past=True)
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (model, decoder_onnx_config_with_past)

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
    models_for_export[ONNX_DECODER_NAME] = (model, onnx_config)

    if config.use_past:
        onnx_config_with_past = config.__class__(model.config, task=config.task, use_past=True)
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (model, onnx_config_with_past)

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
    models_for_export = {}

    # Text encoder
    text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.text_encoder, exporter="onnx", task="feature-extraction"
    )
    text_encoder_onnx_config = text_encoder_config_constructor(pipeline.text_encoder.config)
    models_for_export["text_encoder"] = (pipeline.text_encoder, text_encoder_onnx_config)

    # U-NET
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.unet, exporter="onnx", task="semantic-segmentation", model_type="unet"
    )
    unet_onnx_config = onnx_config_constructor(pipeline.unet.config)

    # PyTorch does not support the ONNX export of torch.nn.functional.scaled_dot_product_attention
    pipeline.unet.set_attn_processor(AttnProcessor())
    models_for_export["unet"] = (pipeline.unet, unet_onnx_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    if not packaging.version.parse(torch.__version__) >= packaging.version.parse("2.1.0"):
        vae_encoder = override_diffusers_2_0_attn_processors(vae_encoder)
    vae_encoder.forward = lambda sample: {"latent_sample": vae_encoder.encode(x=sample)["latent_dist"].sample()}
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder, exporter="onnx", task="semantic-segmentation", model_type="vae-encoder"
    )
    vae_onnx_config = vae_config_constructor(vae_encoder.config)
    models_for_export["vae_encoder"] = (vae_encoder, vae_onnx_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    if not packaging.version.parse(torch.__version__) >= packaging.version.parse("2.1.0"):
        vae_decoder = override_diffusers_2_0_attn_processors(vae_decoder)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder, exporter="onnx", task="semantic-segmentation", model_type="vae-decoder"
    )
    vae_onnx_config = vae_config_constructor(vae_decoder.config)
    models_for_export["vae_decoder"] = (vae_decoder, vae_onnx_config)

    return models_for_export


def override_diffusers_2_0_attn_processors(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, Attention):
            if isinstance(submodule.processor, AttnProcessor2_0):
                submodule.set_processor(AttnProcessor())
            elif isinstance(submodule.processor, LoRAAttnProcessor2_0):
                lora_attn_processor = LoRAAttnProcessor(
                    hidden_size=submodule.processor.hidden_size,
                    cross_attention_dim=submodule.processor.cross_attention_dim,
                    rank=submodule.processor.rank,
                    network_alpha=submodule.processor.to_q_lora.network_alpha,
                )
                lora_attn_processor.to_q_lora = copy.deepcopy(submodule.processor.to_q_lora)
                lora_attn_processor.to_k_lora = copy.deepcopy(submodule.processor.to_k_lora)
                lora_attn_processor.to_v_lora = copy.deepcopy(submodule.processor.to_v_lora)
                lora_attn_processor.to_out_lora = copy.deepcopy(submodule.processor.to_out_lora)
                submodule.set_processor(lora_attn_processor)
            elif isinstance(submodule.processor, AttnAddedKVProcessor2_0):
                submodule.set_processor(AttnAddedKVProcessor())
    return model


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
