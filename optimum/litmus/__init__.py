import copy
import os
import sys
from pathlib import Path
from typing import Dict, List

import onnx
from furiosa.tools.compiler.api import compile
from onnxsim import model_info, onnx_simplifier
from optimum.exporters.onnx import main_export


TARGET_NPU = "warboy-b0"

export_onnx = main_export


def simplify_onnx(input_model: Path, output_model: Path, overwrite_input_shapes: Dict[str, List[int]]) -> None:
    model = onnx.load_model(input_model)
    # https://github.com/daquexian/onnx-simplifier/blob/v0.4.28/onnxsim/onnx_simplifier.py#L479-L521
    print("Simplifying...")
    model_opt, check_ok = onnx_simplifier.simplify(
        model,
        check_n=1,
        overwrite_input_shapes=overwrite_input_shapes,
    )

    try:
        onnx.save(model_opt, output_model)
    except ValueError:
        # large models
        onnx.save(
            copy.deepcopy(model_opt),
            output_model,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(output_model) + ".data",
        )

    if check_ok:
        print("Finish! Here is the difference:")
        model_info.print_simplifying_info(model, model_opt)
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
        )
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(model, model_opt)
        sys.exit(1)


def compile_onnx(input_model: Path, output_dfg: Path, output_dot: Path, target_ir: str = "dfg") -> None:
    onnx_model = onnx.load_model(input_model)
    graph = compile(onnx_model.SerializeToString(), target_ir=target_ir, dot_graph=output_dot, target_npu=TARGET_NPU)
    with open(output_dfg, "wb") as f:
        f.write(bytes(graph))
