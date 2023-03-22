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

from typing import Dict, List

import torch
from transformers import PretrainedConfig


class GgmlConfig:
    def __init__(self, config: "PretrainedConfig", task: str = "default"):
        self.task = task

        self._config = config
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)

    # just a hack to please TasksManager
    @classmethod
    def with_past(cls, *args, **kwargs):
        cls(*args, **kwargs)

    def get_name_map(self, parameters_names: List[str]) -> Dict[str, str]:
        name_map = {}
        for parameter_name in parameters_names:
            name_map[parameter_name] = parameter_name
        return name_map

    def patch_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return state_dict


class TextGgmlConfig(GgmlConfig):
    @property
    def header_data(self):
        return [
            self._normalized_config.vocab_size,
            self._normalized_config.n_positions,
            self._normalized_config.hidden_size,
            self._normalized_config.num_attention_heads,
            self._normalized_config.num_layers,
        ]
