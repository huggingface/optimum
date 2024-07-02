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
"""Optimization with ONNX Runtime command-line interface class."""

from pathlib import Path
from typing import TYPE_CHECKING

from optimum.commands.base import BaseOptimumCLICommand


if TYPE_CHECKING:
    from argparse import ArgumentParser


def parse_args_onnxruntime_optimize(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--onnx_model",
        type=Path,
        required=True,
        help="Path to the repository where the ONNX models to optimize are located.",
    )
    required_group.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the directory where to store generated ONNX model.",
    )

    level_group = parser.add_mutually_exclusive_group(required=True)
    level_group.add_argument(
        "-O1",
        action="store_true",
        help="Basic general optimizations (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details).",
    )
    level_group.add_argument(
        "-O2",
        action="store_true",
        help="Basic and extended general optimizations, transformers-specific fusions (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details).",
    )
    level_group.add_argument(
        "-O3",
        action="store_true",
        help="Same as O2 with Gelu approximation (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details).",
    )
    level_group.add_argument(
        "-O4",
        action="store_true",
        help="Same as O3 with mixed precision (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details).",
    )
    level_group.add_argument(
        "-c",
        "--config",
        type=Path,
        help="`ORTConfig` file to use to optimize the model.",
    )


class ONNXRuntimeOptimizeCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_onnxruntime_optimize(parser)

    def run(self):
        from ...configuration import AutoOptimizationConfig, ORTConfig
        from ...optimization import ORTOptimizer

        if self.args.output == self.args.onnx_model:
            raise ValueError("The output directory must be different than the directory hosting the ONNX model.")

        save_dir = self.args.output

        file_names = [model.name for model in self.args.onnx_model.glob("*.onnx")]
        optimizer = ORTOptimizer.from_pretrained(self.args.onnx_model, file_names)

        if self.args.config:
            optimization_config = ORTConfig.from_pretrained(self.args.config).optimization
        elif self.args.O1:
            optimization_config = AutoOptimizationConfig.O1()
        elif self.args.O2:
            optimization_config = AutoOptimizationConfig.O2()
        elif self.args.O3:
            optimization_config = AutoOptimizationConfig.O3()
        elif self.args.O4:
            optimization_config = AutoOptimizationConfig.O4()
        else:
            raise ValueError("Either -O1, -O2, -O3, -O4 or -c must be specified.")

        optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)
