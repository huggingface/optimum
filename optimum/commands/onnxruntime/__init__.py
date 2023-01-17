import sys
from argparse import ArgumentParser

from .. import BaseOptimumCLICommand
from .optimize import ONNXRuntimmeOptimizeCommand, parse_args_onnxruntime_optimize


def onnxruntime_optimize_factory(args):
    return ONNXRuntimmeOptimizeCommand(args)


class ONNXRuntimeCommand(BaseOptimumCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        onnxruntime_parser = parser.add_parser(
            "onnxruntime", help="ONNX Runtime optimize and quantize utilities."
        )
        onnxruntime_sub_parsers = onnxruntime_parser.add_subparsers()

        optimize_parser = onnxruntime_sub_parsers.add_parser("optimize", help="Optimize ONNX models.")

        parse_args_onnxruntime_optimize(optimize_parser)
        optimize_parser.set_defaults(func=onnxruntime_optimize_factory)

    def run(self):
        raise NotImplementedError()