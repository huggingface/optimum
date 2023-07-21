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
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers import ConfigMixin
from diffusers.image_processor import VaeImageProcessor
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


class OptimumVaeImageProcessor(VaeImageProcessor):
    # Adapted from diffusers.VaeImageProcessor.denormalize
    @staticmethod
    def denormalize(images: np.ndarray):
        """
        Denormalize an image array to [0,1].
        """
        return np.clip(images / 2 + 0.5, 0, 1)

    # Adapted from diffusers.VaeImageProcessor.preprocess
    def preprocess(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """
        Preprocess the image input. Accepted formats are PIL images, NumPy arrays or PyTorch tensors.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)
        if isinstance(image, supported_formats):
            image = [image]
        elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(image[0], PIL.Image.Image):
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            if self.config.do_resize:
                image = [self.resize(i, height, width) for i in image]
            image = self.pil_to_numpy(image)

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            _, _, height, width = image.shape
            if self.config.do_resize and (
                height % self.config.vae_scale_factor != 0 or width % self.config.vae_scale_factor != 0
            ):
                raise ValueError(
                    f"Currently we only support resizing for PIL image - please resize your numpy array to be divisible by {self.config.vae_scale_factor}"
                    f"currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize option in VAEImageProcessor"
                )

        elif isinstance(image[0], torch.Tensor):
            image = (
                torch.cat(image, axis=0).cpu().numpy()
                if image[0].ndim == 4
                else torch.stack(image, axis=0).cpu().numpy()
            )
            _, channel, height, width = image.shape
            # don't need any preprocess if the image is latents
            if channel == 4:
                return image

            if self.config.do_resize and (
                height % self.config.vae_scale_factor != 0 or width % self.config.vae_scale_factor != 0
            ):
                raise ValueError(
                    f"Currently we only support resizing for PIL image - please resize your tensor to be divisible by {self.config.vae_scale_factor}"
                    f"currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize option in VAEImageProcessor"
                )

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize

        if image.min() < 0:
            warnings.warn(
                "Passing `image` as tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.normalize(image)

        return image

    # Adapted from diffusers.VaeImageProcessor.postprocess
    def postprocess(
        self,
        image: np.ndarray,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ):
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support np array"
            )
        if output_type not in ["latent", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            warnings.warn(deprecation_message, FutureWarning)
            output_type = "np"

        if output_type == "latent":
            return image

        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]

        image = np.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])], axis=0
        )
        image = image.transpose((0, 2, 3, 1))

        if output_type == "np":
            return image

        if output_type == "pil":
            return self.numpy_to_pil(image)
