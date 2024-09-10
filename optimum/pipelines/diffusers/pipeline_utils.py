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


import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import ConfigMixin
from tqdm.auto import tqdm
from transformers.modeling_outputs import ModelOutput


logger = logging.getLogger(__name__)


class DiffusionPipelineMixin(ConfigMixin):
    @staticmethod
    def np_to_pt(tensor: np.ndarray, device: str) -> "torch.Tensor":
        return torch.from_numpy(tensor).to(device)

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


def np_randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["np.random.RandomState"], "np.random.RandomState"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    batch_size = shape[0]

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [generator[i].randn(*shape) for i in range(batch_size)]
        latents = np.stack(latents, axis=0)
    elif generator is not None:
        latents = generator.randn(*shape)
    else:
        latents = np.random.randn(*shape)

    return latents


def pt_randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[
        Union[List[Union["torch.Generator", "np.random.RandomState"]], "torch.Generator", "np.random.RandomState"]
    ] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
) -> "torch.Tensor":
    if (isinstance(generator, list) and isinstance(generator[0], torch.Generator)) or isinstance(
        generator, torch.Generator
    ):
        return pt_randn_tensor(shape, generator, device, dtype, layout)
    elif (isinstance(generator, list) and isinstance(generator[0], np.random.RandomState)) or isinstance(
        generator, np.random.RandomState
    ):
        return torch.from_numpy(np_randn_tensor(shape, generator, device, dtype, layout)).to(device)
    else:
        return pt_randn_tensor(shape, generator, device, dtype, layout)


def retrieve_latents(
    encoder_output: ModelOutput, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latent_sample"):
        return encoder_output.latent_sample
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

from contextlib import contextmanager

@contextmanager
def patch_randn_tensor():
    import diffusers.utils.torch_utils

    old_randn_tensor = diffusers.utils.torch_utils.randn_tensor
    diffusers.utils.torch_utils.randn_tensor = randn_tensor
    yield
    diffusers.utils.torch_utils.randn_tensor = old_randn_tensor
