from argparse import ArgumentParser
from pathlib import Path
from transformers.onnx import export, validate_model_outputs
from transformers.onnx.__main__ import check_supported_model_or_raise, get_model_from_features
from transformers import AutoTokenizer


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
        "--features",
        choices=["default"],
        default="default",
        help="Export the model with some additional features.",
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


def convert_to_onnx(model_name_or_path, output, features="default", opset=12):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = get_model_from_features(features, model_name_or_path)
    model_type, model_onnx_config = check_supported_model_or_raise(model, features=features)
    onnx_config = model_onnx_config(model.config)
    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, opset, output)
    return tokenizer, model, onnx_config, onnx_outputs


def main():
    args = parser_export().parse_args()
    args.output = args.output if args.output.suffix else args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    tokenizer, model, onnx_config, onnx_outputs = convert_to_onnx(args.model, args.output, args.features, args.opset)

    validate_model_outputs(onnx_config, tokenizer, model, args.output, onnx_outputs, args.atol)


if __name__ == "__main__":
    main()

