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

from transformers.onnx import validate_model_outputs

from onnxruntime.transformers.fusion_options import FusionOptions

from .convert import convert_to_onnx, parser_export
from .optimize_model import optimize, parser_optimize, quantize


SUPPORTED_MODEL_TYPE = {"bert", "distilbert", "albert", "roberta", "bart", "gpt2"}


def main():
    parser = ArgumentParser(conflict_handler="resolve", parents=[parser_export(), parser_optimize()])
    args = parser.parse_args()

    args.output = args.output if args.output.suffix else args.output.joinpath("model.onnx")
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    tokenizer, model, onnx_config, onnx_outputs = convert_to_onnx(
        args.model_name_or_path, args.output, args.feature, args.opset
    )
    validate_model_outputs(onnx_config, tokenizer, model, args.output, onnx_outputs, atol=args.atol)

    if model.config.model_type not in SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"{model.config.model_type} ({args.model_name_or_path}) is not supported for ONNX Runtime "
            f"optimization. Supported model types are " + ", ".join(SUPPORTED_MODEL_TYPE)
        )

    optimization_options = FusionOptions.parse(args)

    model_type = getattr(model.config, "model_type")
    model_type = "bert" if "bert" in model_type else model_type
    num_heads = getattr(model.config, "num_attention_heads", 0)
    hidden_size = getattr(model.config, "hidden_size", 0)

    args.optimized_output = optimize(
        args.output,
        model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=args.opt_level,
        optimization_options=optimization_options,
        use_gpu=args.use_gpu,
        only_onnxruntime=args.only_onnxruntime,
        use_external_format=args.use_external_format,
    )
    validate_model_outputs(onnx_config, tokenizer, model, args.optimized_output, onnx_outputs, atol=args.atol)

    if args.quantize_dynamic:
        args.quantized_output = quantize(args.optimized_output, use_external_format=args.use_external_format)
        validate_model_outputs(onnx_config, tokenizer, model, args.quantized_output, onnx_outputs, atol=args.atol)


if __name__ == "__main__":
    main()
