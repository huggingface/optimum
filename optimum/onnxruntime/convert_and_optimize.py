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
from .convert import convert_to_onnx, parser_export
from .optimize_model import optimize, quantize, parser_optimize, _get_optimization_options


def main():
    parser = ArgumentParser(conflict_handler='resolve', parents=[parser_export(), parser_optimize()])
    args = parser.parse_args()

    args.output = args.output if args.output.suffix else args.output.joinpath("model.onnx")
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    tokenizer, model, onnx_config, onnx_outputs = convert_to_onnx(args.model, args.output, args.features, args.opset)
    validate_model_outputs(onnx_config, tokenizer, model, args.output, onnx_outputs, atol=args.atol)

    optimization_options = _get_optimization_options(args)

    args.optimized_output = optimize(
        args.output,
        args.model_type,
        args.opt_level,
        optimization_options=optimization_options,
        use_gpu=args.use_gpu,
        only_onnxruntime=args.only_onnxruntime,
        use_external_format=args.use_external_format
    )
    validate_model_outputs(onnx_config, tokenizer, model, args.optimized_output, onnx_outputs, atol=args.atol)

    if args.quantize_dynamic:
        args.quantized_output = quantize(args.optimized_output, use_external_format=args.use_external_format)
        validate_model_outputs(onnx_config, tokenizer, model, args.quantized_output, onnx_outputs, atol=args.atol)


if __name__ == "__main__":
    main()

