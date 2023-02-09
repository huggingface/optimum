from pathlib import Path

from ...onnxruntime.configuration import AutoQuantizationConfig, ORTConfig
from ...onnxruntime.quantization import ORTQuantizer


def parse_args_onnxruntime_quantize(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--onnx_model",
        type=Path,
        required=True,
        help="Path to the repository where the ONNX models to quantize are located.",
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the directory where to store generated ONNX model. (defaults to --onnx_model value).",
    )
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


class ONNXRuntimmeQuantizeCommand:
    def __init__(self, args):
        self.args = args

    def run(self):
        if not self.args.output:
            save_dir = self.args.onnx_model
        else:
            save_dir = self.args.output

        quantizers = []

        quantizers = [
            ORTQuantizer.from_pretrained(save_dir, file_name=model.name)
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
            qconfig = AutoQuantizationConfig.tensorrt(is_static=False, per_channel=self.args.per_channel)
        else:
            qconfig = ORTConfig.from_pretained(self.args.config).quantization

        for q in quantizers:
            q.quantize(save_dir=save_dir, quantization_config=qconfig)
