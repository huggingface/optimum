from argparse import ArgumentParser
from pathlib import Path
from onnxruntime.transformers.optimizer import optimize_model, MODEL_CLASSES
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions


def parse_args(parser=ArgumentParser()):
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        required=True,
        help="Input onnx model path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optimized onnx model path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODEL_CLASSES.keys()),
        default="bert",
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--use_external_format",
        action="store_true",
        help="Allow exporting model >= than 2Gb",
    )
    parser.add_argument(
        "--opt_level",
        type=int,
        choices=[0, 1, 2, 99],
        default=0,
        help="onnxruntime optimization level",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU inference",
    )
    parser.add_argument(
        "--only_onnxruntime",
        action="store_true",
        help="Optimized by onnxruntime only",
    )
    parser.add_argument(
        "--disable_gelu",
        action="store_true",
        help="Disable Gelu fusion",
    )
    parser.add_argument(
        "--disable_layer_norm",
        action="store_true",
        help="Disable LayerNormalization fusion",
    )
    parser.add_argument(
        "--disable_attention",
        action="store_true",
        help="Disable Attention fusion",
    )
    parser.add_argument(
        "--disable_skip_layer_norm",
        action="store_true",
        help="Disable SkipLayerNormalization fusion",
    )
    parser.add_argument(
        "--disable_embed_layer_norm",
        action="store_true",
        help="Disable EmbedLayerNormalization fusion",
    )
    parser.add_argument(
        "--disable_bias_skip_layer_norm",
        action="store_true",
        help="Disable Add Bias and SkipLayerNormalization fusion",
    )
    parser.add_argument(
        "--disable_bias_gelu",
        action="store_true",
        help="Disable Add Bias and Gelu/FastGelu fusion",
    )
    parser.add_argument(
        "--enable_gelu_approximation",
        action="store_true",
        help="Enable Gelu/BiasGelu to FastGelu conversion",
    )
    parser.add_argument(
        "--use_mask_index",
        action="store_true",
        help="Use mask index instead of raw attention mask in attention operator",
    )
    parser.add_argument(
        "--no_attention_mask",
        action="store_true",
        help="No attention mask. Only works for model_type=bert",
    )
    return parser.parse_args()

def _get_optimization_options(args):
    optimization_options = BertOptimizationOptions(args.model_type)
    if args.disable_gelu:
        optimization_options.enable_gelu = False
    if args.disable_layer_norm:
        optimization_options.enable_layer_norm = False
    if args.disable_attention:
        optimization_options.enable_attention = False
    if args.disable_skip_layer_norm:
        optimization_options.enable_skip_layer_norm = False
    if args.disable_embed_layer_norm:
        optimization_options.enable_embed_layer_norm = False
    if args.disable_bias_skip_layer_norm:
        optimization_options.enable_bias_skip_layer_norm = False
    if args.disable_bias_gelu:
        optimization_options.enable_bias_gelu = False
    if args.enable_gelu_approximation:
        optimization_options.enable_gelu_approximation = True
    if args.use_mask_index:
        optimization_options.use_raw_attention_mask(False)
    if args.no_attention_mask:
        optimization_options.disable_attention_mask()
    return optimization_options

def main():
    args = parse_args()
    optimization_options = _get_optimization_options(args)

    optimizer = optimize_model(
        args.onnx_model_path,
        args.model_type,
        opt_level=args.opt_level,
        optimization_options=optimization_options,
        use_gpu=args.use_gpu,
        only_onnxruntime=args.only_onnxruntime,
    )

    optimized_model_filepath = args.onnx_model_path[:-5] + "-optimized.onnx" if not args.output else args.output
    optimizer.save_model_to_file(optimized_model_filepath, args.use_external_format)

    if optimizer.is_fully_optimized():
        print("The model has been fully optimized.")
    else:
        print("The model has been optimized.")

if __name__ == "__main__":
    main()
















