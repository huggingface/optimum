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
from diffusers.image_processor import VaeImageProcessor as DiffusersVaeImageProcessor
from diffusers.utils.pil_utils import PIL_INTERPOLATION
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


class VaeImageProcessor(DiffusersVaeImageProcessor):
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

        do_convert_grayscale = getattr(self.config, "do_convert_grayscale", False)
        # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
        if do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                # if image is a pytorch tensor could have 2 possible shapes:
                #    1. batch x height x width: we should insert the channel dimension at position 1
                #    2. channnel x height x width: we should insert batch dimension at position 0,
                #       however, since both channel and batch dimension has same size 1, it is same to insert at position 1
                #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
                image = image.unsqueeze(1)
            else:
                # if it is a numpy array, it could have 2 possible shapes:
                #   1. batch x height x width: insert channel dimension on last position
                #   2. height x width x channel: insert batch dimension on first position
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, supported_formats):
            image = [image]
        elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(image[0], PIL.Image.Image):
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            if self.config.do_resize:
                height, width = self.get_height_width(image[0], height, width)
                image = [self.resize(i, height, width) for i in image]
            image = self.reshape(self.pil_to_numpy(image))
        else:
            if isinstance(image[0], torch.Tensor):
                image = [self.pt_to_numpy(elem) for elem in image]
                image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            else:
                image = self.reshape(np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0))

            if do_convert_grayscale and image.ndim == 3:
                image = np.expand_dims(image, 1)

            # don't need any preprocess if the image is latents
            if image.shape[1] == 4:
                return image

            if self.config.do_resize:
                height, width = self.get_height_width(image, height, width)
                image = self.resize(image, height, width)

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if image.min() < 0 and do_normalize:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.normalize(image)

        if getattr(self.config, "do_binarize", False):
            image = self.binarize(image)

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

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return image

    def get_height_width(
        self,
        image: [PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        """
        This function return the height and width that are downscaled to the next integer multiple of
        `vae_scale_factor`.

        Args:
            image(`PIL.Image.Image`, `np.ndarray`):
                The image input, can be a PIL image, numpy array or pytorch tensor. if it is a numpy array, should have
                shape `[batch, height, width]` or `[batch, height, width, channel]` if it is a pytorch tensor, should
                have shape `[batch, channel, height, width]`.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed image. If `None`, will use the height of `image` input.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed. If `None`, will use the width of the `image` input.
        """
        height = height or (image.height if isinstance(image, PIL.Image.Image) else image.shape[-2])
        width = width or (image.width if isinstance(image, PIL.Image.Image) else image.shape[-1])
        # resize to integer multiple of vae_scale_factor
        width, height = (x - x % self.config.vae_scale_factor for x in (width, height))
        return height, width

    # Adapted from diffusers.VaeImageProcessor.numpy_to_pt
    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images)
        return images

    # Adapted from diffusers.VaeImageProcessor.pt_to_numpy
    @staticmethod
    def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().float().numpy()
        return images

    @staticmethod
    def reshape(images: np.ndarray) -> np.ndarray:
        """
        Reshape inputs to expected shape.
        """
        if images.ndim == 3:
            images = images[..., None]

        return images.transpose(0, 3, 1, 2)

    # TODO : remove after diffusers v0.21.0 release
    def resize(
        self,
        image: [PIL.Image.Image, np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> [PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        Resize image.
        """
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), resample=PIL_INTERPOLATION[self.config.resample])
        elif isinstance(image, torch.Tensor):
            image = torch.nn.functional.interpolate(image, size=(height, width))
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pt(image)
            image = torch.nn.functional.interpolate(image, size=(height, width))
            image = self.pt_to_numpy(image)
        return image
