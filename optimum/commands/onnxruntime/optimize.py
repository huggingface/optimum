from pathlib import Path
from argparse import ArgumentParser

from ...onnxruntime.optimization import ORTOptimizer
from ...onnxruntime.configuration import AutoOptimizationConfig


def parse_args_onnxruntime_optimize(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--onnx_model", type=Path, required=True, help="Path indicating where the ONNX models to optimize are located."
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-o", "--output", type=Path, help="Path indicating the directory where to store generated ONNX model. (defaults to --onnx_model value)."
    )
    
    level_group = parser.add_mutually_exclusive_group(required=True)
    level_group.add_argument(
        "-O1", action="store_true", help="Basic general optimizations (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details)."
    )
    level_group.add_argument(
        "-O2", action="store_true", help="Basic and extended general optimizations, transformers-specific fusions (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details)."
    )
    level_group.add_argument(
        "-O3", action="store_true", help="Same as O2 with Gelu approximation (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details)."
    )
    level_group.add_argument(
        "-O4", action="store_true", help="Same as O3 with mixed precision (see: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization for more details)."
    )

class ONNXRuntimmeOptimizeCommand:
    def __init__(self, args):
        self.args = args

    def run(self):
        if not self.args.output:
            save_dir = self.args.onnx_model
        else:
            save_dir = self.args.output
        
        file_names = [model.name for model in save_dir.glob("*.onnx")]
        
        optimizer = ORTOptimizer.from_pretrained(self.args.onnx_model, file_names)

        if self.args.O1:
            optimization_config = AutoOptimizationConfig.O1()
        elif self.args.O2:
            optimization_config = AutoOptimizationConfig.O2()
        elif self.args.O3:
            optimization_config = AutoOptimizationConfig.O3()
        else:
            optimization_config = AutoOptimizationConfig.O4()
        
        optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)
