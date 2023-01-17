import sys
from argparse import ArgumentParser

from .. import BaseOptimumCLICommand
from .optimize import ONNXRuntimmeOptimizeCommand, parse_args_onnxruntime_optimize
from .quantize import ONNXRuntimmeQuantizeCommand, parse_args_onnxruntime_quantize


def onnxruntime_optimize_factory(args):
    return ONNXRuntimmeOptimizeCommand(args)


def onnxruntime_quantize_factory(args):
    return ONNXRuntimmeQuantizeCommand(args)


class ONNXRuntimeCommand(BaseOptimumCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        onnxruntime_parser = parser.add_parser("onnxruntime", help="ONNX Runtime optimize and quantize utilities.")
        onnxruntime_sub_parsers = onnxruntime_parser.add_subparsers()

        optimize_parser = onnxruntime_sub_parsers.add_parser("optimize", help="Optimize ONNX models.")
        quantize_parser = onnxruntime_sub_parsers.add_parser("quantize", help="Dynammic quantization for ONNX models.")

        parse_args_onnxruntime_optimize(optimize_parser)
        parse_args_onnxruntime_quantize(quantize_parser)

        optimize_parser.set_defaults(func=onnxruntime_optimize_factory)
        quantize_parser.set_defaults(func=onnxruntime_quantize_factory)

    def run(self):
        raise NotImplementedError()
