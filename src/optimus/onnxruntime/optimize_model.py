import os
from argparse import ArgumentParser
from onnxruntime.transformers.optimizer import MODEL_CLASSES
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions


def parser(parser=ArgumentParser()):
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        help="Input ONNX model path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optimized ONNX model path.",
    )
    parser.add_argument(
        "--type_model",
        type=str,
        choices=list(MODEL_CLASSES.keys()),
        default="bert",
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--use_external_format",
        action="store_true",
        help="Allow exporting model >= than 2Gb.",
    )
    parser.add_argument(
        "--opt_level",
        type=int,
        choices=[0, 1, 2, 99],
        default=0,
        help="ONNX Runtime optimization level.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU inference.",
    )
    parser.add_argument(
        "--only_onnxruntime",
        action="store_true",
        help="Optimized by ONNX Runtime only.",
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

def _get_optimization_options(args):
    optimization_options = BertOptimizationOptions(args.type_model)
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

def optimize(
        onnx_model_path,
        type_model,
        opt_level,
        output,
        optimization_options=None,
        use_gpu=False,
        only_onnxruntime=False,
        use_external_format=False
):

    from onnxruntime.transformers.optimizer import optimize_model

    optimizer = optimize_model(
        onnx_model_path,
        type_model,
        opt_level=opt_level,
        optimization_options=optimization_options,
        use_gpu=use_gpu,
        only_onnxruntime=only_onnxruntime,
    )

    optimizer.save_model_to_file(output, use_external_format)

    if optimizer.is_fully_optimized():
        print("The model has been fully optimized.")
    else:
        print("The model has been optimized.")


def main():
    args = parser().parse_args()
    optimization_options = _get_optimization_options(args)

    if args.onnx_model_path is None or not os.path.exists(args.onnx_model_path) or not args.onnx_model_path.endswith(".onnx"):
        raise Exception("Invalid ONNX model path.")

    output = args.onnx_model_path[:-5] + "-optimized.onnx" if args.output is None else args.output

    optimize(
        args.onnx_model_path,
        args.type_model,
        args.opt_level,
        output=output,
        optimization_options=optimization_options,
        use_gpu=args.use_gpu,
        only_onnxruntime=args.only_onnxruntime,
        use_external_format=args.use_external_format
    )


if __name__ == "__main__":
    main()







