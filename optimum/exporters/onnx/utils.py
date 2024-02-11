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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5HifiGan
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


MODEL_TYPES_REQUIRING_POSITION_IDS = {
    "codegen",
    "falcon",
    "gpt2",
    "gpt-bigcode",
    "gpt-neo",
    "gpt-neox",
    "gptj",
    "imagegpt",
    "llama",
    "phi",
    "mistral",
}


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


def _get_submodels_for_export_stable_diffusion(
    pipeline: "StableDiffusionPipeline",
) -> Dict[str, Union["PreTrainedModel", "ModelMixin"]]:
    """
    Returns the components of a Stable Diffusion model.
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline

    models_for_export = {}
    if isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
        projection_dim = pipeline.text_encoder_2.config.projection_dim
    else:
        projection_dim = pipeline.text_encoder.config.projection_dim

    # Text encoder
    if pipeline.text_encoder is not None:
        if isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
            pipeline.text_encoder.config.output_hidden_states = True
        models_for_export["text_encoder"] = pipeline.text_encoder

    # U-NET
    # ONNX export of torch.nn.functional.scaled_dot_product_attention not supported for < v2.1.0
    is_torch_greater_or_equal_than_2_1 = version.parse(torch.__version__) >= version.parse("2.1.0")
    if not is_torch_greater_or_equal_than_2_1:
        pipeline.unet.set_attn_processor(AttnProcessor())
    pipeline.unet.config.text_encoder_projection_dim = projection_dim
    # The U-NET time_ids inputs shapes depends on the value of `requires_aesthetics_score`
    # https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L571
    pipeline.unet.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)
    models_for_export["unet"] = pipeline.unet

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    if not is_torch_greater_or_equal_than_2_1:
        vae_encoder = override_diffusers_2_0_attn_processors(vae_encoder)
    vae_encoder.forward = lambda sample: {"latent_sample": vae_encoder.encode(x=sample)["latent_dist"].sample()}
    models_for_export["vae_encoder"] = vae_encoder

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    if not is_torch_greater_or_equal_than_2_1:
        vae_decoder = override_diffusers_2_0_attn_processors(vae_decoder)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    models_for_export["vae_decoder"] = vae_decoder

    text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
    if text_encoder_2 is not None:
        text_encoder_2.config.output_hidden_states = True
        models_for_export["text_encoder_2"] = text_encoder_2

    return models_for_export


def _get_submodels_for_export_decoder(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    use_past: bool,
    legacy: bool = False,
) -> Dict[str, Union["PreTrainedModel", "TFPreTrainedModel"]]:
    """
    Returns the decoder part of the model.
    """
    models_for_export = {ONNX_DECODER_NAME if legacy else "model": model}

    if legacy and use_past:
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = model

    return models_for_export


def _get_submodels_for_export_encoder_decoder(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], use_past: bool
) -> Dict[str, Union["PreTrainedModel", "TFPreTrainedModel"]]:
    """
    Returns the encoder and decoder parts of the model.
    """
    models_for_export = {}

    encoder_model = model.get_encoder()
    models_for_export[ONNX_ENCODER_NAME] = encoder_model
    models_for_export[ONNX_DECODER_NAME] = model
    if use_past:
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = model

    return models_for_export


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
    models_for_export = _get_submodels_for_export_encoder_decoder(model, use_past=config.use_past)

    encoder_onnx_config = config.with_behavior("encoder")
    models_for_export[ONNX_ENCODER_NAME] = (models_for_export[ONNX_ENCODER_NAME], encoder_onnx_config)

    decoder_onnx_config = config.with_behavior("decoder", use_past=config.use_past, use_past_in_inputs=False)
    models_for_export[ONNX_DECODER_NAME] = (models_for_export[ONNX_DECODER_NAME], decoder_onnx_config)

    if config.use_past:
        decoder_onnx_config_with_past = config.with_behavior("decoder", use_past=True, use_past_in_inputs=True)
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (
            models_for_export[ONNX_DECODER_WITH_PAST_NAME],
            decoder_onnx_config_with_past,
        )

    return models_for_export


def get_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    config: "OnnxConfig",
    legacy: bool = False,
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

    models_for_export = _get_submodels_for_export_decoder(model, use_past=config.use_past, legacy=legacy)

    onnx_kwargs = {
        "task": config.task,
        "float_dtype": config.float_dtype,
        "int_dtype": config.int_dtype,
        "legacy": legacy,
    }

    if legacy:
        onnx_config = config.__class__(
            model.config,
            use_past=config.use_past,
            use_past_in_inputs=False,
            **onnx_kwargs,
        )
        models_for_export[ONNX_DECODER_NAME] = (models_for_export[ONNX_DECODER_NAME], onnx_config)

        if config.use_past:
            onnx_config_with_past = config.__class__(
                model.config,
                use_past=True,
                use_past_in_inputs=True,
                **onnx_kwargs,
            )
            models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (
                models_for_export[ONNX_DECODER_WITH_PAST_NAME],
                onnx_config_with_past,
            )

    else:
        onnx_config = config.__class__(
            model.config,
            use_past=config.use_past,
            use_past_in_inputs=config.use_past,
            **onnx_kwargs,
        )
        models_for_export["model"] = (models_for_export["model"], onnx_config)

    return models_for_export


def get_stable_diffusion_models_for_export(
    pipeline: "StableDiffusionPipeline",
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "OnnxConfig"]]:
    """
    Returns the components of a Stable Diffusion model and their subsequent onnx configs.

    Args:
        pipeline ([`StableDiffusionPipeline`]):
            The model to export.
        int_dtype (`str`, defaults to `"int64"`):
            The data type of integer tensors, could be ["int64", "int32", "int8"], default to "int64".
        float_dtype (`str`, defaults to `"fp32"`):
            The data type of float tensors, could be ["fp32", "fp16", "bf16"], default to "fp32".

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]: A Dict containing the model and
        onnx configs for the different components of the model.
    """
    models_for_export = _get_submodels_for_export_stable_diffusion(pipeline)

    # Text encoder
    if "text_encoder" in models_for_export:
        text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
            model=pipeline.text_encoder,
            exporter="onnx",
            library_name="diffusers",
            task="feature-extraction",
        )
        text_encoder_onnx_config = text_encoder_config_constructor(
            pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
        )
        models_for_export["text_encoder"] = (models_for_export["text_encoder"], text_encoder_onnx_config)

    # U-NET
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.unet,
        exporter="onnx",
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="unet",
    )
    unet_onnx_config = onnx_config_constructor(pipeline.unet.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["unet"] = (models_for_export["unet"], unet_onnx_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = models_for_export["vae_encoder"]
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter="onnx",
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-encoder",
    )
    vae_onnx_config = vae_config_constructor(vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["vae_encoder"] = (vae_encoder, vae_onnx_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = models_for_export["vae_decoder"]
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter="onnx",
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-decoder",
    )
    vae_onnx_config = vae_config_constructor(vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["vae_decoder"] = (vae_decoder, vae_onnx_config)

    if "text_encoder_2" in models_for_export:
        onnx_config_constructor = TasksManager.get_exporter_config_constructor(
            model=pipeline.text_encoder_2,
            exporter="onnx",
            library_name="diffusers",
            task="feature-extraction",
            model_type="clip-text-with-projection",
        )
        onnx_config = onnx_config_constructor(
            pipeline.text_encoder_2.config, int_dtype=int_dtype, float_dtype=float_dtype
        )
        models_for_export["text_encoder_2"] = (models_for_export["text_encoder_2"], onnx_config)

    return models_for_export


def _get_submodels_for_export_sam(model, variant):
    models_for_export = {}

    if variant == "monolith":
        models_for_export["model"] = model
    else:
        # We rather use the model patcher to patch their forward method.
        models_for_export["vision_encoder"] = model
        models_for_export["prompt_encoder_mask_decoder"] = model

    return models_for_export


def get_sam_models_for_export(model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "OnnxConfig"):
    models_for_export = _get_submodels_for_export_sam(model, config.variant)

    if config.variant == "monolith":
        onnx_config = config.__class__(model.config, task=config.task, legacy=config.legacy)
        models_for_export["model"] = (models_for_export["model"], onnx_config)
    else:
        vision_encoder_onnx_config = config.__class__(
            model.config, task=config.task, variant=config.variant, vision_encoder=True, legacy=config.legacy
        )
        prompt_encoder_mask_decoder_onnx_config = config.__class__(
            model.config, task=config.task, variant=config.variant, vision_encoder=False, legacy=config.legacy
        )
        models_for_export["vision_encoder"] = (models_for_export["vision_encoder"], vision_encoder_onnx_config)
        models_for_export["prompt_encoder_mask_decoder"] = (
            models_for_export["prompt_encoder_mask_decoder"],
            prompt_encoder_mask_decoder_onnx_config,
        )

    return models_for_export


def get_speecht5_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "OnnxConfig", model_kwargs: Optional[Dict]
):
    if model_kwargs is None or "vocoder" not in model_kwargs:
        raise ValueError(
            'The ONNX export of SpeechT5 requires a vocoder. Please pass `--model-kwargs \'{"vocoder": "vocoder_model_name_or_path"}\'` from the command line, or `model_kwargs={"vocoder": "vocoder_model_name_or_path"}` if calling main_export.'
        )

    models_for_export = {}

    # We rather use the model patcher to patch their forward method.
    models_for_export["encoder_model"] = model
    models_for_export["decoder_model"] = model

    if config.variant == "with-past":
        models_for_export["decoder_with_past_model"] = model

    # TODO: more flexibility in the vocoder class?
    vocoder = SpeechT5HifiGan.from_pretrained(model_kwargs["vocoder"]).eval()
    model_kwargs["vocoder_model"] = vocoder

    models_for_export["decoder_postnet_and_vocoder"] = model

    encoder_onnx_config = config.with_behavior("encoder")

    use_past = config.variant == "with-past"
    decoder_onnx_config = config.with_behavior("decoder", use_past=use_past, use_past_in_inputs=False)

    models_for_export[ONNX_ENCODER_NAME] = (models_for_export[ONNX_ENCODER_NAME], encoder_onnx_config)
    models_for_export[ONNX_DECODER_NAME] = (models_for_export[ONNX_DECODER_NAME], decoder_onnx_config)
    if config.variant == "with-past":
        decoder_onnx_config_with_past = config.with_behavior("decoder", use_past=True, use_past_in_inputs=True)
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (
            models_for_export[ONNX_DECODER_WITH_PAST_NAME],
            decoder_onnx_config_with_past,
        )

    postnet_and_vocoder_onnx_config = config.__class__(
        config._config,
        task=config.task,
        int_dtype=config.int_dtype,
        float_dtype=config.float_dtype,
        use_past=use_past,
        use_past_in_inputs=False,  # Irrelevant here.
        behavior=config._behavior,  # Irrelevant here.
        preprocessors=config._preprocessors,
        is_postnet_and_vocoder=True,
        legacy=config.legacy,
    )
    postnet_and_vocoder_onnx_config.variant = config.variant
    models_for_export["decoder_postnet_and_vocoder"] = (
        models_for_export["decoder_postnet_and_vocoder"],
        postnet_and_vocoder_onnx_config,
    )

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
    if not custom_architecture:
        if library_name == "diffusers":
            onnx_config = None
            models_and_onnx_configs = get_stable_diffusion_models_for_export(
                model, int_dtype=int_dtype, float_dtype=float_dtype
            )
        else:
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                model=model, exporter="onnx", task=task, library_name=library_name
            )
            onnx_config = onnx_config_constructor(
                model.config,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                preprocessors=preprocessors,
                legacy=legacy,
            )

            onnx_config.variant = _variant
            all_variants = "\n".join(
                [f"    - {name}: {description}" for name, description in onnx_config.VARIANTS.items()]
            )
            logger.info(f"Using the export variant {onnx_config.variant}. Available variants are:\n{all_variants}")

            # TODO: this succession of if/else strongly suggests a refactor is needed.
            if (
                model.config.is_encoder_decoder
                and task.startswith(TasksManager._ENCODER_DECODER_TASKS)
                and not monolith
            ):
                models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
            elif task.startswith("text-generation") and not monolith:
                models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config, legacy=legacy)
            elif model.config.model_type == "sam":
                models_and_onnx_configs = get_sam_models_for_export(model, onnx_config)
            elif model.config.model_type == "speecht5":
                models_and_onnx_configs = get_speecht5_models_for_export(model, onnx_config, model_kwargs)
            else:
                models_and_onnx_configs = {"model": (model, onnx_config)}

        # When specifying custom ONNX configs for supported transformers architectures, we do
        # not force to specify a custom ONNX config for each submodel.
        for key, custom_onnx_config in custom_onnx_configs.items():
            models_and_onnx_configs[key] = (models_and_onnx_configs[key][0], custom_onnx_config)
    else:
        onnx_config = None
        submodels_for_export = None
        models_and_onnx_configs = {}

        if fn_get_submodels is not None:
            submodels_for_export = fn_get_submodels(model)
        else:
            if library_name == "diffusers":
                submodels_for_export = _get_submodels_for_export_stable_diffusion(model)
            elif (
                model.config.is_encoder_decoder
                and task.startswith(TasksManager._ENCODER_DECODER_TASKS)
                and not monolith
            ):
                submodels_for_export = _get_submodels_for_export_encoder_decoder(
                    model, use_past=task.endswith("-with-past")
                )
            elif task.startswith("text-generation") and not monolith:
                submodels_for_export = _get_submodels_for_export_decoder(model, use_past=task.endswith("-with-past"))
            else:
                submodels_for_export = {"model": model}

        if submodels_for_export.keys() != custom_onnx_configs.keys():
            logger.error(f"ONNX custom configs for: {', '.join(custom_onnx_configs.keys())}")
            logger.error(f"Submodels to export: {', '.join(submodels_for_export.keys())}")
            raise ValueError(
                "Trying to export a custom model, but could not find as many custom ONNX configs as the number of submodels to export. Please specifiy the fn_get_submodels argument, that should return a dictionary of submodules with as many items as the provided custom_onnx_configs dictionary."
            )

        for key, custom_onnx_config in custom_onnx_configs.items():
            models_and_onnx_configs[key] = (submodels_for_export[key], custom_onnx_config)

    # Default to the first ONNX config for stable-diffusion and custom architecture case.
    if onnx_config is None:
        onnx_config = next(iter(models_and_onnx_configs.values()))[1]

    return onnx_config, models_and_onnx_configs
