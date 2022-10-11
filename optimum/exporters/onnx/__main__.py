# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entry point to the optimum.exporters.onnx command line."""

from argparse import ArgumentParser
from pathlib import Path

from ...utils import logging
from ..features import FeaturesManager
from .convert import export, validate_model_outputs


def main():
    parser = ArgumentParser("Hugging Face Optimum ONNX exporter")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    parser.add_argument(
        "--feature",
        default="default",
        help="The type of features to export the model with.",
    )
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version to export the model with.")
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute difference tolerence when validating the model."
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["pt", "tf"],
        default=None,
        help=(
            "The framework to use for the ONNX export."
            " If not provided, will attempt to use the local checkpoint's original framework"
            " or what is available in the environment."
        ),
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    parser.add_argument("output", type=Path, help="Path indicating where to store generated ONNX model.")

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output if args.output.is_file() else args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    # Allocate the model
    model = FeaturesManager.get_model_from_feature(
        args.feature, args.model, framework=args.framework, cache_dir=args.cache_dir
    )
    onnx_config_constructor = FeaturesManager.get_exporter_config_constructor(model, "onnx", feature=args.feature)
    onnx_config = onnx_config_constructor(model.config)

    # Ensure the requested opset is sufficient
    if args.opset is None:
        args.opset = onnx_config.DEFAULT_ONNX_OPSET

    if args.opset < onnx_config.DEFAULT_ONNX_OPSET:
        raise ValueError(
            f"Opset {args.opset} is not sufficient to export {model.config.model_type}. "
            f"At least  {onnx_config.DEFAULT_ONNX_OPSET} is required."
        )

    onnx_inputs, onnx_outputs = export(
        model,
        onnx_config,
        args.opset,
        args.output,
    )

    if args.atol is None:
        args.atol = onnx_config.atol_for_validation

    validate_model_outputs(onnx_config, model, args.output, onnx_outputs, args.atol)
    logger.info(f"All good, model saved at: {args.output.as_posix()}")


if __name__ == "__main__":
    logger = logging.get_logger()  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    main()
