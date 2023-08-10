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
"""ggml configuration base classes."""

from abc import ABC

from ..base import ExportConfig


class GgmlConfig(ExportConfig, ABC):
    """
    Base class for GGML exportable model.
    """

    def __init__(self, config: "PretrainedConfig", task: str = "feature-extraction"):
        self.task = task
        self._config = config


class GgmlConfigWithPast(GgmlConfig, ABC):
    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "feature-extraction") -> "OnnxConfigWithPast":
        """
        Instantiates a [`~optimum.exporters.onnx.OnnxConfig`] with `use_past` attribute set to `True`.

        Args:
            config (`transformers.PretrainedConfig`):
                The underlying model's config to use when exporting to ONNX.
            task (`str`, defaults to `"feature-extraction"`):
                The task the model should be exported for.

        Returns:
            [`~optimum.exporters.onnx.GgmlConfig`]: The ggml config with `.use_past = True`
        """
        return cls(config, task=task, use_past=True)
