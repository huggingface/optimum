# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Defines the command line for the export with ONNX."""

from typing import TYPE_CHECKING

from ...exporters.onnx.__main__ import main_export, parse_args_onnx
from ...utils import DEFAULT_DUMMY_SHAPES
from ..base import BaseOptimumCLICommand


if TYPE_CHECKING:
    from argparse import ArgumentParser


class ONNXExportCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_onnx(parser)

    def run(self):
        # Get the shapes to be used to generate dummy inputs
        input_shapes = {}
        for input_name in DEFAULT_DUMMY_SHAPES.keys():
            input_shapes[input_name] = getattr(self.args, input_name)

        main_export(
            model_name_or_path=self.args.model,
            output=self.args.output,
            task=self.args.task,
            opset=self.args.opset,
            device=self.args.device,
            fp16=self.args.fp16,
            optimize=self.args.optimize,
            monolith=self.args.monolith,
            no_post_process=self.args.no_post_process,
            framework=self.args.framework,
            atol=self.args.atol,
            cache_dir=self.args.cache_dir,
            trust_remote_code=self.args.trust_remote_code,
            pad_token_id=self.args.pad_token_id,
            for_ort=self.args.for_ort,
            **input_shapes,
        )
