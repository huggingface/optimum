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

import sys
from argparse import ArgumentParser, RawTextHelpFormatter

from ...exporters.onnx.__main__ import parse_args_onnx
from .. import BaseOptimumCLICommand
from .onnx import ONNXExportCommand
from .tflite import TFLiteExportCommand, parse_args_tflite


def onnx_export_factory(args):
    return ONNXExportCommand(args)


def tflite_export_factory(_):
    return TFLiteExportCommand(" ".join(sys.argv[3:]))


class ExportCommand(BaseOptimumCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        export_parser = parser.add_parser(
            "export", help="Export PyTorch and TensorFlow models to several format (currently supported: onnx)."
        )
        export_sub_parsers = export_parser.add_subparsers()

        onnx_parser = export_sub_parsers.add_parser(
            "onnx", help="Export PyTorch and TensorFlow to ONNX.", formatter_class=RawTextHelpFormatter
        )

        parse_args_onnx(onnx_parser)
        onnx_parser.set_defaults(func=onnx_export_factory)

        tflite_parser = export_sub_parsers.add_parser("tflite", help="Export TensorFlow to TensorFlow Lite.")

        parse_args_tflite(tflite_parser)
        tflite_parser.set_defaults(func=tflite_export_factory)

    def run(self):
        raise NotImplementedError()
