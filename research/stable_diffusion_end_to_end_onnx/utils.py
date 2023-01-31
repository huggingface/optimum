from PIL import Image

from optimum.exporters.onnx.convert import check_dummy_inputs_are_allowed
import torch
import torch.nn as nn

from typing import Dict, Tuple, Union, List, Optional

import logging

logger = logging.getLogger()


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


class StableDiffusionPreprocessor:
    def __init__(
        self,
        tokenizer,
        text_encoder_config,
        do_classifier_free_guidance: bool,
        scheduler,
    ):
        self.text_encoder_config = text_encoder_config
        self.tokenizer = tokenizer
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.scheduler = scheduler

    def preprocess(self, prompt: str, num_inference_steps: int = 50, device: str = "cpu"):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        result = {"text_input_ids": text_input_ids, "timesteps": timesteps}

        if self.do_classifier_free_guidance:
            batch_size = text_input_ids.shape[0]
            max_length = text_input_ids.shape[1]

            uncond_tokens = [""] * batch_size
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            result["uncond_text_input_ids"] = uncond_input.input_ids

        return result


def export_models(
    models_and_onnx_configs: Dict[str, Tuple["PreTrainedModel", "OnnxConfig"]],
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Exports a Pytorch or TensorFlow encoder decoder model to an ONNX Intermediate Representation.
    The following method exports the encoder and decoder components of the model as separate
    ONNX files.

    Args:
        models_and_onnx_configs (`Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]]):
            A dictionnary containing the models to export and their corresponding onnx configs.
            If None, will use the keys from `models_and_onnx_configs` as names.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
    Returns:
        `Tuple[List[List[str]], List[List[str]]]`: A tuple with an ordered list of the model's inputs, and the named
        inputs from the ONNX configuration.
    """
    outputs = {}

    for i, model_name in enumerate(models_and_onnx_configs.keys()):
        submodel, sub_onnx_config = models_and_onnx_configs[model_name]

        outputs[model_name] = export_pytorch(
            model=submodel,
            config=sub_onnx_config,
            device=device,
            input_shapes=input_shapes,
        )

    return outputs


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: "OnnxConfig",
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to an ONNX Intermediate Representation.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    from torch.utils._pytree import tree_map

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
        if device.type == "cuda" and torch.cuda.is_available():
            model.to(device)
            dummy_inputs = tree_map(
                lambda value: value.to(device) if isinstance(value, torch.Tensor) else value, dummy_inputs
            )
        check_dummy_inputs_are_allowed(model, dummy_inputs)
        inputs = config.ordered_inputs(model)
        input_names = list(inputs.keys())
        output_names = list(config.outputs.keys())

        return input_names, dummy_inputs, output_names

def place_inputs_on_device(inputs: Tuple[torch.Tensor], device: torch.device):
    inputs = list(inputs)
    for i in range(len(inputs)):
        inputs[i] = inputs[i].to(device)
    return tuple(inputs)

def get_traced_submodules(in_dummy_out_per_model, models_and_onnx_configs):
    print("Tracing text encoder")
    dummy_inputs = tuple(in_dummy_out_per_model["text_encoder"][1].values())
    submodel = models_and_onnx_configs["text_encoder"][0].eval()
    dummy_inputs = place_inputs_on_device(dummy_inputs, submodel.device)

    # torch.inference_mode is not strong enough
    for param in submodel.parameters():
        param.requires_grad = False

    # submodel_modif = lambda x: submodel(x, return_dict=False)
    submodel.config.return_dict = False
    text_encoder_traced = torch.jit.trace(submodel, dummy_inputs)

    print("Tracing unet")
    dummy_inputs = tuple(in_dummy_out_per_model["unet"][1].values())
    submodel = models_and_onnx_configs["unet"][0].eval()
    dummy_inputs = place_inputs_on_device(dummy_inputs, submodel.device)

    for param in submodel.parameters():
        param.requires_grad = False

    # submodel_modif = lambda x, y, z: submodel(x, y, z, return_dict=False)
    if submodel.forward.__defaults__[3] is True:
        defaults = list(submodel.forward.__defaults__)
        defaults[3] = False  # return_dict = False
        submodel.forward.__func__.__defaults__ = tuple(defaults)
    unet_traced = torch.jit.trace(submodel, dummy_inputs)

    print("Tracing vae decoder")
    dummy_inputs = tuple(in_dummy_out_per_model["vae_decoder"][1].values())
    vae_decoder = models_and_onnx_configs["vae_decoder"][0].eval()
    dummy_inputs = place_inputs_on_device(dummy_inputs, submodel.device)

    for param in vae_decoder.parameters():
        param.requires_grad = False

    class VAEDecoderForward(nn.Module):
        def __init__(self, vae_decoder):
            super().__init__()

            self.vae_decoder = vae_decoder

        def forward(self, latent_sample):
            return self.vae_decoder.decode(z=latent_sample, return_dict=False)

    vae_decoder_module = VAEDecoderForward(vae_decoder)

    vae_decoder_traced = torch.jit.trace(vae_decoder_module, dummy_inputs)

    return text_encoder_traced, unet_traced, vae_decoder_traced
