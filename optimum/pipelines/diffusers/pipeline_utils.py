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
from typing import Union

import numpy as np
import torch
from diffusers import ConfigMixin
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


class DiffusionPipelineMixin(ConfigMixin):
    @staticmethod
    def np_to_pt(
        np_object: Union[np.ndarray, np.random.RandomState], device: str
    ) -> Union[torch.Tensor, torch.Generator]:
        if isinstance(np_object, np.ndarray):
            return torch.from_numpy(np_object).to(device)
        elif isinstance(np_object, np.random.RandomState):
            return torch.Generator(device=device).manual_seed(int(np_object.get_state()[1][0]))
        else:
            raise ValueError(f"Unsupported type {type(np_object)}")

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
