# coding=utf-8
#  Copyright 2022 The HuggingFace Team. All rights reserved.
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
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class NormalizedConfig:
    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "hidden_size"
    NUM_LAYERS = "num_hidden_layers"
    NUM_ATTENTION_HEADS = "num_attention_heads"

    def __init__(self, config: "PretrainedConfig"):
        self.config = config

    def __getattr__(self, attr_name):
        if attr_name.startswith("__"):
            return getattr(self, attr_name)
        if attr_name.upper() not in self.__class__.__dict__:
            raise AttributeError(f"Could not find the attribute named \"{attr_name.upper()}\" in the normalized config.")
        return getattr(self.config, getattr(self, attr_name.upper()))


# TODO: make it framework agnostic
class DummyInputGenerator(ABC):
    SUPPORTED_INPUT_NAMES = ()

    def supports_input(self, input_name: str) -> bool:
        return input_name in self.SUPPORTED_INPUT_NAMES

    @abstractmethod
    def generate(self, input_name: str, framework: Optional[str] = "pt") -> torch.Tensor:
        raise NotImplementedError

    # def for_task(cls, task: str) -> "DummyInputGenerator":
    #     return cls(task=task)

    def random_int_tensor(self, shape: List[int], max_value: int, min_value: int = 0, framework: Optional[str] = "pt") -> torch.Tensor:
        return torch.randint(low=min_value, high=max_value, size=shape)


class DummyTextInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
    )

    def __init__(self, task: str, normalized_config: NormalizedConfig, batch_size: Optional[int] = None, sequence_length: Optional[int] = None, num_choices: Optional[int] = None):
        self.task = task
        self.vocab_size = normalized_config.vocab_size
        self.batch_size = random.randint(2, 4) if batch_size is None else batch_size
        self.sequence_length = random.randint(128, 384) if sequence_length is None else sequence_length
        self.num_choices = random.randint(2, 4) if num_choices is None else num_choices

    def generate(self, input_name: str, framework: Optional[str] = "pt") -> torch.Tensor:
        min_value = 0
        max_value = 2 if input_name != "input_ids" else self.vocab_size
        shape = [self.batch_size, self.sequence_length]
        if self.task == "multiple_choice":
            # TODO: check that.
            shape = [self.num_choices, self.batch_size, self.sequence_length]
        return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework)
