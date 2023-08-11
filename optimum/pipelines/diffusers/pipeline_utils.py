#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import warnings

import numpy as np
import PIL
import torch
from diffusers import ConfigMixin
from diffusers.utils import PIL_INTERPOLATION
from PIL import Image
from tqdm.auto import tqdm


class DiffusionPipelineMixin(ConfigMixin):
    # Copied from https://github.com/huggingface/diffusers/blob/v0.12.1/src/diffusers/pipelines/pipeline_utils.py#L812
    @staticmethod
    def numpy_to_pil(images):
        """
        Converts a numpy image or a batch of images to a PIL image.
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

    # Copied from https://github.com/huggingface/diffusers/blob/v0.12.1/src/diffusers/pipelines/pipeline_utils.py#L827
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")


# Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess with 8->64
def preprocess(image):
    warnings.warn(
        (
            "The preprocess method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor.preprocess instead"
        ),
        FutureWarning,
    )
    if isinstance(image, torch.Tensor):
        return image.cpu().numpy()
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0).cpu().numpy()
    return image


# Adapted from https://github.com/huggingface/diffusers/blob/v0.18.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L58
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = np.std(noise_pred_text, axis=tuple(range(1, noise_pred_text.ndim)), keepdims=True)
    std_cfg = np.std(noise_cfg, axis=tuple(range(1, noise_cfg.ndim)), keepdims=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
