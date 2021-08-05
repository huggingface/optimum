from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoTokenizer
from transformers.utils import check_min_version
from transformers.onnx import export, validate_model_outputs

check_min_version("4.10.0.dev0")

from transformers.onnx.features import FeaturesManager


def parser_export(parser=None):
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model's id or path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path indicating where to store generated ONNX model.",
    )
    parser.add_argument(
        "--feature",
        choices=list(FeaturesManager.AVAILABLE_FEATURES),
        default="default",
        help="Export the model with some additional feature.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version to export the model with.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute difference tolerance when validating the model.",
    )
    return parser


def convert_to_onnx(model_name_or_path, output, feature="default", opset=12):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FeaturesManager.get_model_from_feature(feature, model_name_or_path)
    model_type, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    onnx_config = model_onnx_config(model.config)
    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, opset, output)
    return tokenizer, model, onnx_config, onnx_outputs


def main():
    args = parser_export().parse_args()
    args.output = args.output if args.output.suffix else args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    tokenizer, model, onnx_config, onnx_outputs = convert_to_onnx(args.model, args.output, args.feature, args.opset)

    validate_model_outputs(onnx_config, tokenizer, model, args.output, onnx_outputs, args.atol)


if __name__ == "__main__":
    main()

