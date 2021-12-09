#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from argparse import ArgumentParser
from pathlib import Path

import onnx


try:
    from onnxruntime.transformers.optimizer import MODEL_TYPES
except ImportError:
    from onnxruntime.transformers.optimizer import MODEL_CLASSES as MODEL_TYPES

from typing import Optional

from onnxruntime.transformers.fusion_options import FusionOptions

from .utils import generate_identified_filename


def parser_optimize(parser=None):
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument(
        "--onnx_model_path",
        type=Path,
        help="Input ONNX model path.",
    )
    parser.add_argument(
        "--opt_level",
        type=int,
        choices=[0, 1, 2, 99],
        default=0,
        help="Optimization level performed by ONNX Runtime of the loaded graph."
        "0 will disable all optimizations."
        "1 will enable basic optimizations."
        "2 will enable basic and extended optimizations, including complex node fusions applied to the nodes "
        "assigned to the CPU or CUDA execution provider, making the resulting optimized graph hardware dependent."
        "99 will enable all available optimizations including layout optimizations.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODEL_TYPES.keys()),
        default="bert",
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES.keys()),
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=0,
        help="Number of attention heads. 12 for bert-base model and 16 for bert-large."
        "For bert model_type, 0 allows to detect the parameter from graph automatically",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=0,
        help="Model hidden size. 768 for bert-base model and 1024 for bert-large."
        "For bert model_type, 0 allows to detect the parameter from graph automatically",
    )
    parser.add_argument(
        "--only_onnxruntime",
        action="store_true",
        help="Whether to only use ONNX Runtime to optimize model and no graph fusion in Python.",
    )
    parser.add_argument(
        "--quantize_dynamic",
        action="store_true",
        help="Apply dynamic quantization.",
    )
    parser.add_argument(
        "--use_external_format",
        action="store_true",
        help="Allow exporting model >= than 2Gb.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Whether to optimize the model for GPU inference."
        "The optimized graph might contain operators for GPU or CPU only when opt_level > 1.",
    )
    parser.add_argument(
        "--disable_gelu",
        action="store_true",
        help="Disable Gelu fusion.",
    )
    parser.add_argument(
        "--disable_layer_norm",
        action="store_true",
        help="Disable LayerNormalization fusion.",
    )
    parser.add_argument(
        "--disable_attention",
        action="store_true",
        help="Disable Attention fusion.",
    )
    parser.add_argument(
        "--disable_skip_layer_norm",
        action="store_true",
        help="Disable SkipLayerNormalization fusion.",
    )
    parser.add_argument(
        "--disable_embed_layer_norm",
        action="store_true",
        help="Disable EmbedLayerNormalization fusion.",
    )
    parser.add_argument(
        "--disable_bias_skip_layer_norm",
        action="store_true",
        help="Disable Add Bias and SkipLayerNormalization fusion.",
    )
    parser.add_argument(
        "--disable_bias_gelu",
        action="store_true",
        help="Disable Add Bias and Gelu/FastGelu fusion.",
    )
    parser.add_argument(
        "--enable_gelu_approximation",
        action="store_true",
        help="Enable Gelu/BiasGelu to FastGelu conversion.",
    )
    parser.add_argument(
        "--use_mask_index",
        action="store_true",
        help="Use mask index instead of raw attention mask in attention operator.",
    )
    parser.add_argument(
        "--no_attention_mask",
        action="store_true",
        help="No attention mask. Only works for model_type=bert.",
    )
    return parser


def optimize(
    onnx_model_path: Path,
    model_type: str,
    num_heads: int = 0,
    hidden_size: int = 0,
    opt_level: Optional[int] = None,
    optimization_options: Optional[FusionOptions] = None,
    use_gpu: bool = False,
    only_onnxruntime: bool = False,
    use_external_format: bool = False,
):
    """
    Given an ONNX model, create an optimized ONNX model and save it.

    Args:
        onnx_model_path (:obj:`Path`):
            Path indicating the ONNX model to optimize.
        model_type (:obj:`str`):
            Model type used to obtain the optimizer class, the default opt_level and export tools.
        num_heads (:obj:`int`):
            Number of attention heads. For model_type bert, 0 allows to detect the parameter from graph automatically.
        hidden_size (:obj:`int`):
            Model hidden size. For model_type bert, 0 allows to detect the parameter from graph automatically.
        opt_level (:obj:`int`, `optional`):
            Optimization level performed by ONNX Runtime of the loaded graph.
            0 will disable all optimizations.
            1 will enable basic optimizations.
            2 will enable basic and extended optimizations, including complex node fusions applied to the nodes
            assigned to the CPU or CUDA execution provider, making the resulting optimized graph hardware dependent.
            99 will enable all available optimizations including layout optimizations.
        optimization_options (:obj:`FusionOptions`, `optional`):
            Optimization options used to turn on or off the different fusion options.
        use_gpu (:obj:`bool`):
            Whether to optimize the model for GPU inference. The optimized graph might contain operators for GPU or CPU
            only when opt_level > 1.
        only_onnxruntime (:obj:`bool`):
            Whether to only use ONNX Runtime to optimize model and no graph fusion in Python.
        use_external_format (:obj:`bool`):
            Allow exporting model >= than 2Gb.
    """
    from onnxruntime.transformers.optimizer import optimize_model

    optimizer = optimize_model(
        onnx_model_path.as_posix(),
        model_type,
        num_heads,
        hidden_size,
        opt_level=opt_level,
        optimization_options=optimization_options,
        use_gpu=use_gpu,
        only_onnxruntime=only_onnxruntime,
    )

    optimized_model_path = generate_identified_filename(onnx_model_path, "-optimized")
    optimizer.save_model_to_file(optimized_model_path.as_posix(), use_external_format)
    print(f"Optimized model saved to: {optimized_model_path}")

    if optimizer.is_fully_optimized():
        print("The model has been fully optimized.")
    else:
        print("The model has been optimized.")

    return optimized_model_path


def quantize(onnx_model_path: Path, optimize_model: bool = False, use_external_format: bool = False):
    """
    Given an ONNX model, create a quantized ONNX model and save it.

    Args:
        onnx_model_path (:obj:`Path`):
            Path indicating the ONNX model to quantize.
        optimize_model (:obj:`bool`):
            Whether to optimize the model before quantization.
        use_external_format (:obj:`bool`):
            Allow exporting model >= than 2Gb.
    """
    from onnxruntime.quantization import QuantType, onnx_model, quantize_dynamic

    model = onnx.load(onnx_model_path.as_posix())
    model = onnx_model.ONNXModel(model)
    model.replace_gemm_with_matmul()
    tmp_model_path = generate_identified_filename(onnx_model_path, "-tmp")
    onnx.save_model(model.model, tmp_model_path.as_posix())
    quantized_model_path = generate_identified_filename(onnx_model_path, "-quantized")

    quantize_dynamic(
        tmp_model_path,
        quantized_model_path,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        optimize_model=optimize_model,
        use_external_data_format=use_external_format,
    )

    print(f"Quantized model saved to: {quantized_model_path}")
    os.remove(tmp_model_path)

    return quantized_model_path


def main():
    args = parser_optimize().parse_args()

    if args.onnx_model_path is None or not args.onnx_model_path.is_file():
        raise Exception("Invalid ONNX model path.")

    optimization_options = FusionOptions.parse(args)

    args.optimized_output = optimize(
        args.onnx_model_path,
        args.model_type,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        opt_level=args.opt_level,
        optimization_options=optimization_options,
        use_gpu=args.use_gpu,
        only_onnxruntime=args.only_onnxruntime,
        use_external_format=args.use_external_format,
    )

    if args.quantize_dynamic:
        args.quantized_output = quantize(args.optimized_output, use_external_format=args.use_external_format)


if __name__ == "__main__":
    main()
