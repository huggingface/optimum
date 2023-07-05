import argparse
from pathlib import Path

from optimum.litmus import compile_onnx, export_onnx, utils
from optimum.litmus.nlp import BATCH_SIZE, INPUT_LENGTH, TASKS, simplify


def main():
    parser = argparse.ArgumentParser("FuriosaAI litmus OPT using HF Optimum API.")
    parser.add_argument("output_dir", type=Path, help="path to directory to save outputs")
    parser.add_argument(
        "--model-size",
        "-s",
        choices=["125m", "350m", "1.3b", "2.7b", "6.7b", "30b", "66b"],
        help="available model sizes",
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

    model_name = f"opt-{args.model_size}"
    model_tag = f"facebook/{model_name}"
    output_dir = args.output_dir

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if not (output_dir / "decoder_model_merged.onnx").exists():
        print("Exporting ONNX Model...")
        export_onnx(model_name_or_path=model_tag, output=output_dir, task=args.task, framework="pt")

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
