import argparse
from pathlib import Path

from optimum.litmus import compile_onnx, export_onnx, simplify_onnx
from optimum.litmus.nlp import BATCH_SIZE, SEQUENCE_LENGTH, TASKS


def main():
    parser = argparse.ArgumentParser("FuriosaAI litmus GPT Neo using HF Optimum API.")
    parser.add_argument("output_dir", help="path to directory to save outputs")
    parser.add_argument("--size", "-s", choices=["125m", "1.3b", "2.7b"], help="availible model sizes")
    args = parser.parse_args()

    model_name = f"gpt-neo-{args.size}"
    task = "text-generation"
    assert task in TASKS
    output = Path(args.output_dir)
    if not output.exists():
        output.mkdir(parents=True)
    export_onnx(model_name_or_path=f"EleutherAI/{model_name}", output=output, task=task, framework="pt")

    onnx_model_name = "decoder_model"
    input_model = output / f"{onnx_model_name}.onnx"
    opt_model = output / f"{onnx_model_name}-opt.onnx"
    overwrite_input_shapes = {
        "input_ids": [BATCH_SIZE, SEQUENCE_LENGTH],
        "attention_mask": [BATCH_SIZE, SEQUENCE_LENGTH],
    }
    simplify_onnx(input_model, opt_model, overwrite_input_shapes)

    compile_onnx(opt_model, output / f"{model_name}.dfg", output / f"{model_name}.dot")


if __name__ == "__main__":
    main()
