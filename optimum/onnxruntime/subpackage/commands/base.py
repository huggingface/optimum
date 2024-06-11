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
"""optimum.onnxruntime command-line interface base classes."""

from optimum.commands import BaseOptimumCLICommand, CommandInfo, optimum_cli_subcommand

from .optimize import ONNXRuntimeOptimizeCommand
from .quantize import ONNXRuntimeQuantizeCommand


@optimum_cli_subcommand()
class ONNXRuntimeCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(
        name="onnxruntime",
        help="ONNX Runtime optimize and quantize utilities.",
    )
    SUBCOMMANDS = (
        CommandInfo(
            name="optimize",
            help="Optimize ONNX models.",
            subcommand_class=ONNXRuntimeOptimizeCommand,
        ),
        CommandInfo(
            name="quantize",
            help="Dynammic quantization for ONNX models.",
            subcommand_class=ONNXRuntimeQuantizeCommand,
        ),
    )
