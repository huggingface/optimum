import argparse
import json
from pathlib import Path

from rich import print_json

import optimum
from optimum.litmus import compile_onnx, export_toy_onnx, utils
from optimum.litmus.nlp import BATCH_SIZE, INPUT_LENGTH, TASKS, simplify


def main():
    parser = argparse.ArgumentParser(
        "FuriosaAI litmus exporting toy model(w/o pretrained weights) using HF Optimum API."
    )
    parser.add_argument("output_dir", type=Path, help="path to directory to save outputs")
    parser.add_argument(
        "--config-path", "-c", type=str, help="path to model config saved in json format"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        default=BATCH_SIZE,
        type=utils.check_non_negative,
        help="Batch size for model inputs",
    )
    parser.add_argument(
        "--input-len",
        default=INPUT_LENGTH,
        type=utils.check_non_negative,
        help="Length of input prommpt",
    )
    parser.add_argument(
        "--gen-step",
        default=0,
        type=utils.check_non_negative,
        help="Generation step to simplify onnx graph",
    )
    parser.add_argument(
        "--task",
        default="text-generation-with-past",
        type=str,
        choices=TASKS,
        help="Task to export model for",
    )
    args = parser.parse_args()

    json_path = args.config_path
    output_dir = args.output_dir
    print("Proceeding model exporting and optimization based given model config:")
    with open(json_path, "r") as f:
        print_json(data=json.load(f))

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if not (output_dir / "decoder_model_merged.onnx").exists():
        print("Exporting ONNX Model...")
        export_toy_onnx(json_path, output=output_dir, task=args.task, framework="pt")

    print("Simplifying ONNX Model...")
    simplify(
        output_dir / "decoder_model_merged.onnx",
        output_dir,
        args.batch_size,
        args.input_len,
        args.gen_step,
        args.task,
    )

    compile_onnx(
        output_dir / f"decoder_model-opt_gen_step={args.gen_step}.onnx",
        output_dir / f"decoder_model-opt_gen_step={args.gen_step}.dfg",
        output_dir / f"decoder_model-opt_gen_step={args.gen_step}.dot",
    )


if __name__ == "__main__":
    main()
