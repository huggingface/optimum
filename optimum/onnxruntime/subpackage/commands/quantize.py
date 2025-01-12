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
"""Quantization with ONNX Runtime command-line interface class."""

from pathlib import Path
from typing import TYPE_CHECKING

from optimum.commands import BaseOptimumCLICommand


if TYPE_CHECKING:
    from argparse import ArgumentParser


def parse_args_onnxruntime_quantize(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--onnx_model",
        type=Path,
        required=True,
        help="Path to the repository where the ONNX models to quantize are located.",
    )
    required_group.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the directory where to store generated ONNX model.",
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--per_channel",
        action="store_true",
        help="Compute the quantization parameters on a per-channel basis.",
    )

    level_group = parser.add_mutually_exclusive_group(required=True)
    level_group.add_argument("--arm64", action="store_true", help="Quantization for the ARM64 architecture.")
    level_group.add_argument("--avx2", action="store_true", help="Quantization with AVX-2 instructions.")
    level_group.add_argument("--avx512", action="store_true", help="Quantization with AVX-512 instructions.")
    level_group.add_argument(
        "--avx512_vnni", action="store_true", help="Quantization with AVX-512 and VNNI instructions."
    )
    level_group.add_argument("--tensorrt", action="store_true", help="Quantization for NVIDIA TensorRT optimizer.")
    level_group.add_argument(
        "-c",
        "--config",
        type=Path,
        help="`ORTConfig` file to use to optimize the model.",
    )


class ONNXRuntimeQuantizeCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_onnxruntime_quantize(parser)

    def run(self):
        from ...configuration import AutoQuantizationConfig, ORTConfig
        from ...quantization import ORTQuantizer

        if self.args.output == self.args.onnx_model:
            raise ValueError("The output directory must be different than the directory hosting the ONNX model.")

        save_dir = self.args.output
        quantizers = []
        use_external_data_format = False

        quantizers = [
            ORTQuantizer.from_pretrained(self.args.onnx_model, file_name=model.name)
            for model in self.args.onnx_model.glob("*.onnx")
        ]

        if self.args.arm64:
            qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=self.args.per_channel)
        elif self.args.avx2:
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=self.args.per_channel)
        elif self.args.avx512:
            qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=self.args.per_channel)
        elif self.args.avx512_vnni:
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=self.args.per_channel)
        elif self.args.tensorrt:
            raise ValueError(
                "TensorRT quantization relies on static quantization that requires calibration, which is currently not supported through optimum-cli. Please adapt Optimum static quantization examples to run static quantization for TensorRT: https://github.com/huggingface/optimum/tree/main/examples/onnxruntime/quantization"
            )
        else:
            config = ORTConfig.from_pretrained(self.args.config)
            qconfig = config.quantization
            use_external_data_format = config.use_external_data_format

        for q in quantizers:
            q.quantize(
                save_dir=save_dir, quantization_config=qconfig, use_external_data_format=use_external_data_format
            )
