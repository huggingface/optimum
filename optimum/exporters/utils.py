# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

"""Utilities for model preparation to export."""


import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5HifiGan
from transformers.utils import is_tf_available, is_torch_available

from ..utils import (
    DIFFUSERS_MINIMUM_VERSION,
    check_if_diffusers_greater,
    is_diffusers_available,
    logging,
)
from ..utils.import_utils import _diffusers_version
from .tasks import TasksManager


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
    from .base import ExportConfig

    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin, StableDiffusionPipeline


ENCODER_NAME = "encoder_model"
DECODER_NAME = "decoder_model"
DECODER_WITH_PAST_NAME = "decoder_with_past_model"
DECODER_MERGED_NAME = "decoder_model_merged"


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
    models_for_export = {DECODER_NAME if legacy else "model": model}

    if legacy and use_past:
        models_for_export[DECODER_WITH_PAST_NAME] = model

    return models_for_export


def _get_submodels_for_export_encoder_decoder(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], use_past: bool
) -> Dict[str, Union["PreTrainedModel", "TFPreTrainedModel"]]:
    """
    Returns the encoder and decoder parts of the model.
    """
    models_for_export = {}

    encoder_model = model.get_encoder()
    models_for_export[ENCODER_NAME] = encoder_model
    models_for_export[DECODER_NAME] = model
    if use_past:
        models_for_export[DECODER_WITH_PAST_NAME] = model

    return models_for_export


def get_encoder_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "ExportConfig"
) -> Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel"], "ExportConfig"]]:
    """
    Returns the encoder and decoder parts of the model and their subsequent export configs.

    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.base.ExportConfig`]):
            The export configuration associated with the exported model.

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `ExportConfig`]: A Dict containing the model and
        export configs for the encoder and decoder parts of the model.
    """
    models_for_export = _get_submodels_for_export_encoder_decoder(model, use_past=config.use_past)

    encoder_export_config = config.with_behavior("encoder")
    models_for_export[ENCODER_NAME] = (models_for_export[ENCODER_NAME], encoder_export_config)

    decoder_export_config = config.with_behavior("decoder", use_past=config.use_past, use_past_in_inputs=False)
    models_for_export[DECODER_NAME] = (models_for_export[DECODER_NAME], decoder_export_config)

    if config.use_past:
        decoder_export_config_with_past = config.with_behavior("decoder", use_past=True, use_past_in_inputs=True)
        models_for_export[DECODER_WITH_PAST_NAME] = (
            models_for_export[DECODER_WITH_PAST_NAME],
            decoder_export_config_with_past,
        )

    return models_for_export


def get_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    config: "ExportConfig",
    legacy: bool = False,
) -> Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel"], "ExportConfig"]]:
    """
    Returns two versions of the decoder that can be used together to perform fast generation:

        1. The first one takes regular inputs, and outputs the result along with past key/values.
        2. The second one takes regular inputs and past key/values, and outputs the result along with the updated past
        key/values.


    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.base.ExportConfig`]):
            The export configuration associated with the exported model.

    Returns:
        `Dict[str, Tuple[Union[PreTrainedModel, TFPreTrainedModel], ExportConfig]]: A Dict containing the model and
        export configs for the encoder and decoder parts of the model.
    """

    models_for_export = _get_submodels_for_export_decoder(model, use_past=config.use_past, legacy=legacy)

    export_kwargs = {
        "task": config.task,
        "float_dtype": config.float_dtype,
        "int_dtype": config.int_dtype,
        "legacy": legacy,
    }

    if legacy:
        export_config = config.__class__(
            model.config,
            use_past=config.use_past,
            use_past_in_inputs=False,
            **export_kwargs,
        )
        models_for_export[DECODER_NAME] = (models_for_export[DECODER_NAME], export_config)

        if config.use_past:
            export_config_with_past = config.__class__(
                model.config,
                use_past=True,
                use_past_in_inputs=True,
                **export_kwargs,
            )
            models_for_export[DECODER_WITH_PAST_NAME] = (
                models_for_export[DECODER_WITH_PAST_NAME],
                export_config_with_past,
            )

    else:
        export_config = config.__class__(
            model.config,
            use_past=config.use_past,
            use_past_in_inputs=config.use_past,
            **export_kwargs,
        )
        models_for_export["model"] = (models_for_export["model"], export_config)

    return models_for_export


def get_stable_diffusion_models_for_export(
    pipeline: "StableDiffusionPipeline",
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    exporter: str = "onnx",
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "ExportConfig"]]:
    """
    Returns the components of a Stable Diffusion model and their subsequent export configs.

    Args:
        pipeline ([`StableDiffusionPipeline`]):
            The model to export.
        int_dtype (`str`, defaults to `"int64"`):
            The data type of integer tensors, could be ["int64", "int32", "int8"], default to "int64".
        float_dtype (`str`, defaults to `"fp32"`):
            The data type of float tensors, could be ["fp32", "fp16", "bf16"], default to "fp32".

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `ExportConfig`]: A Dict containing the model and
        export configs for the different components of the model.
    """
    models_for_export = _get_submodels_for_export_stable_diffusion(pipeline)

    # Text encoder
    if "text_encoder" in models_for_export:
        text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
            model=pipeline.text_encoder,
            exporter=exporter,
            library_name="diffusers",
            task="feature-extraction",
        )
        text_encoder_export_config = text_encoder_config_constructor(
            pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
        )
        models_for_export["text_encoder"] = (models_for_export["text_encoder"], text_encoder_export_config)

    # U-NET
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.unet,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="unet",
    )
    unet_export_config = export_config_constructor(pipeline.unet.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["unet"] = (models_for_export["unet"], unet_export_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = models_for_export["vae_encoder"]
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-encoder",
    )
    vae_export_config = vae_config_constructor(vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["vae_encoder"] = (vae_encoder, vae_export_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = models_for_export["vae_decoder"]
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-decoder",
    )
    vae_export_config = vae_config_constructor(vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["vae_decoder"] = (vae_decoder, vae_export_config)

    if "text_encoder_2" in models_for_export:
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=pipeline.text_encoder_2,
            exporter=exporter,
            library_name="diffusers",
            task="feature-extraction",
            model_type="clip-text-with-projection",
        )
        export_config = export_config_constructor(
            pipeline.text_encoder_2.config, int_dtype=int_dtype, float_dtype=float_dtype
        )
        models_for_export["text_encoder_2"] = (models_for_export["text_encoder_2"], export_config)

    return models_for_export


def get_musicgen_models_for_export(model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "ExportConfig"):
    models_for_export = {
        "text_encoder": model.text_encoder,
        "encodec_decode": model.audio_encoder,
        # For the decoder, we do not pass model.decoder because we may need to export model.enc_to_dec_proj
        DECODER_NAME: model,
        DECODER_WITH_PAST_NAME: model,
        "build_delay_pattern_mask": model.decoder,
    }

    text_encoder_config = config.__class__(
        model.config, task=config.task, legacy=False, model_part="text_encoder", variant=config.variant
    )
    models_for_export["text_encoder"] = (models_for_export["text_encoder"], text_encoder_config)

    audio_encoder_config = config.__class__(
        model.config, task=config.task, legacy=False, model_part="encodec_decode", variant=config.variant
    )
    models_for_export["encodec_decode"] = (models_for_export["encodec_decode"], audio_encoder_config)

    use_past = "with-past" in config.variant
    decoder_export_config = config.with_behavior("decoder", use_past=use_past, use_past_in_inputs=False)
    decoder_export_config.model_part = "decoder"
    models_for_export[DECODER_NAME] = (models_for_export[DECODER_NAME], decoder_export_config)

    if "with-past" in config.variant:
        decoder_export_config_with_past = config.with_behavior("decoder", use_past=True, use_past_in_inputs=True)
        decoder_export_config_with_past.model_part = "decoder"
        models_for_export[DECODER_WITH_PAST_NAME] = (
            models_for_export[DECODER_WITH_PAST_NAME],
            decoder_export_config_with_past,
        )

    build_delay_pattern_mask_config = config.__class__(
        model.config, task=config.task, legacy=False, model_part="build_delay_pattern_mask", variant=config.variant
    )
    models_for_export["build_delay_pattern_mask"] = (
        models_for_export["build_delay_pattern_mask"],
        build_delay_pattern_mask_config,
    )

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


def get_sam_models_for_export(model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "ExportConfig"):
    models_for_export = _get_submodels_for_export_sam(model, config.variant)

    if config.variant == "monolith":
        export_config = config.__class__(model.config, task=config.task, legacy=config.legacy)
        models_for_export["model"] = (models_for_export["model"], export_config)
    else:
        vision_encoder_export_config = config.__class__(
            model.config, task=config.task, variant=config.variant, vision_encoder=True, legacy=config.legacy
        )
        prompt_encoder_mask_decoder_export_config = config.__class__(
            model.config, task=config.task, variant=config.variant, vision_encoder=False, legacy=config.legacy
        )
        models_for_export["vision_encoder"] = (models_for_export["vision_encoder"], vision_encoder_export_config)
        models_for_export["prompt_encoder_mask_decoder"] = (
            models_for_export["prompt_encoder_mask_decoder"],
            prompt_encoder_mask_decoder_export_config,
        )

    return models_for_export


def get_speecht5_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "ExportConfig", model_kwargs: Optional[Dict]
):
    if model_kwargs is None or "vocoder" not in model_kwargs:
        raise ValueError(
            'The export of SpeechT5 requires a vocoder. Please pass `--model-kwargs \'{"vocoder": "vocoder_model_name_or_path"}\'` from the command line, or `model_kwargs={"vocoder": "vocoder_model_name_or_path"}` if calling main_export.'
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

    encoder_export_config = config.with_behavior("encoder")

    use_past = config.variant == "with-past"
    decoder_export_config = config.with_behavior("decoder", use_past=use_past, use_past_in_inputs=False)

    models_for_export[ENCODER_NAME] = (models_for_export[ENCODER_NAME], encoder_export_config)
    models_for_export[DECODER_NAME] = (models_for_export[DECODER_NAME], decoder_export_config)
    if config.variant == "with-past":
        decoder_export_config_with_past = config.with_behavior("decoder", use_past=True, use_past_in_inputs=True)
        models_for_export[DECODER_WITH_PAST_NAME] = (
            models_for_export[DECODER_WITH_PAST_NAME],
            decoder_export_config_with_past,
        )

    postnet_and_vocoder_export_config = config.__class__(
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
    postnet_and_vocoder_export_config.variant = config.variant
    models_for_export["decoder_postnet_and_vocoder"] = (
        models_for_export["decoder_postnet_and_vocoder"],
        postnet_and_vocoder_export_config,
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


def _get_submodels_and_export_configs(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    task: str,
    monolith: bool,
    custom_export_configs: Dict,
    custom_architecture: bool,
    _variant: str,
    library_name: str,
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    fn_get_submodels: Optional[Callable] = None,
    preprocessors: Optional[List[Any]] = None,
    legacy: bool = False,
    model_kwargs: Optional[Dict] = None,
    exporter: str = "onnx",
):
    if not custom_architecture:
        if library_name == "diffusers":
            export_config = None
            models_and_export_configs = get_stable_diffusion_models_for_export(
                model, int_dtype=int_dtype, float_dtype=float_dtype, exporter=exporter
            )
        else:
            export_config_constructor = TasksManager.get_exporter_config_constructor(
                model=model, exporter=exporter, task=task, library_name=library_name
            )
            export_config = export_config_constructor(
                model.config,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                preprocessors=preprocessors,
                legacy=legacy,
            )

            export_config.variant = _variant
            all_variants = "\n".join(
                [f"    - {name}: {description}" for name, description in export_config.VARIANTS.items()]
            )
            logger.info(f"Using the export variant {export_config.variant}. Available variants are:\n{all_variants}")

            # TODO: this succession of if/else strongly suggests a refactor is needed.
            if (
                model.config.is_encoder_decoder
                and task.startswith(TasksManager._ENCODER_DECODER_TASKS)
                and not monolith
            ):
                models_and_export_configs = get_encoder_decoder_models_for_export(model, export_config)
            elif task.startswith("text-generation") and not monolith:
                models_and_export_configs = get_decoder_models_for_export(model, export_config, legacy=legacy)
            elif model.config.model_type == "sam":
                models_and_export_configs = get_sam_models_for_export(model, export_config)
            elif model.config.model_type == "speecht5":
                models_and_export_configs = get_speecht5_models_for_export(model, export_config, model_kwargs)
            elif model.config.model_type == "musicgen":
                models_and_export_configs = get_musicgen_models_for_export(model, export_config)
            else:
                models_and_export_configs = {"model": (model, export_config)}

        # When specifying custom export configs for supported transformers architectures, we do
        # not force to specify a custom export config for each submodel.
        for key, custom_export_config in custom_export_configs.items():
            models_and_export_configs[key] = (models_and_export_configs[key][0], custom_export_config)
    else:
        export_config = None
        submodels_for_export = None
        models_and_export_configs = {}

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

        if submodels_for_export.keys() != custom_export_configs.keys():
            logger.error(f"{exporter.upper()} custom configs for: {', '.join(custom_export_configs.keys())}")
            logger.error(f"Submodels to export: {', '.join(submodels_for_export.keys())}")
            raise ValueError(
                f"Trying to export a custom model, but could not find as many custom {exporter.upper()} configs as the number of submodels to export. Please specifiy the fn_get_submodels argument, that should return a dictionary of submodules with as many items as the provided custom_export_configs dictionary."
            )

        for key, custom_export_config in custom_export_configs.items():
            models_and_export_configs[key] = (submodels_for_export[key], custom_export_config)

    # Default to the first ONNX config for stable-diffusion and custom architecture case.
    if export_config is None:
        export_config = next(iter(models_and_export_configs.values()))[1]

    return export_config, models_and_export_configs
