import argparse
from pathlib import Path

from optimum.litmus import compile_onnx, export_onnx, simplify_onnx
from optimum.litmus.nlp import BATCH_SIZE, SEQUENCE_LENGTH, TASKS


def main():
    parser = argparse.ArgumentParser("FuriosaAI litmus LLaMA using HF Optimum API.")
    parser.add_argument("output_dir", help="path to directory to save outputs")
    parser.add_argument("--size", "-s", choices=["7b", "13b", "30b", "65b"], help="available model sizes")
    args = parser.parse_args()

    model_name = f"llama-{args.size}"
    model_tag = f"decapoda-research/{model_name}-hf"
    task = "text-generation-with-past"
    assert task in TASKS
    output = Path(args.output_dir)
    if not output.exists():
        output.mkdir(parents=True)
    export_onnx(model_name_or_path=model_tag, output=output, task=task, framework="pt")

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
