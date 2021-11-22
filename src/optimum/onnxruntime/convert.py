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

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from transformers.onnx import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager


def parser_export(parser=None):
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to model or model identifier from huggingface.co/models.",
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
        default=None,
        help="ONNX opset version to export the model with.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute difference tolerance when validating the model.",
    )
    return parser


def convert_to_onnx(model_name_or_path: str, output: Path, feature: str = "default", opset: Optional[int] = None):
    """
    Load and export a model to an ONNX Intermediate Representation (IR).

    Args:
        model_name_or_path (:obj:`str`):
            Repository name in the Hugging Face Hub or path to a local directory containing the model to export.
        output (:obj:`Path`):
            Path indicating where to store the generated ONNX model.
        feature (:obj:`str`):
            Export the model with some additional feature.
        opset (:obj:`int`, `optional`):
            Define the ONNX opset version used to export the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FeaturesManager.get_model_from_feature(feature, model_name_or_path)
    model_type, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    onnx_config = model_onnx_config(model.config)
    opset = onnx_config.default_onnx_opset if opset is None else opset
    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, opset, output)
    return tokenizer, model, onnx_config, onnx_outputs


def main():
    args = parser_export().parse_args()
    args.output = args.output if args.output.suffix else args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    tokenizer, model, onnx_config, onnx_outputs = convert_to_onnx(
        args.model_name_or_path, args.output, args.feature, args.opset
    )

    validate_model_outputs(onnx_config, tokenizer, model, args.output, onnx_outputs, args.atol)


if __name__ == "__main__":
    main()
