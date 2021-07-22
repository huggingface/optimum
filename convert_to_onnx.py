from argparse import ArgumentParser
import argparse
from pathlib import Path
from transformers.convert_graph_to_onnx import convert, SUPPORTED_PIPELINES


def parse_args(parser=ArgumentParser()):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model's id or path (ex: bert-base-cased)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="onnx model path",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer's id or path (ex: bert-base-cased)",
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["pt", "tf"],
        help="Framework for loading the model",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=SUPPORTED_PIPELINES,
        default="feature-extraction",
        help="Pipeline from the list: " + ", ".join(SUPPORTED_PIPELINES),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="onnx opset to use",
    )
    parser.add_argument(
        "--use_external_format",
        action="store_true",
        help="Allow exporting model >= than 2Gb",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output = Path(args.output).absolute()

    try:
        convert(
            framework=args.framework,
            model=args.model,
            output=args.output,
            opset=args.opset,
            tokenizer=args.tokenizer,
            use_external_format=args.use_external_format,
            pipeline_name=args.pipeline,
        )
    except Exception as e:
        print(f"Error while converting the model: {e}")


















