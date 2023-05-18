import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import onnx
import onnxruntime as ort
from furiosa.tools.compiler.api import compile
from onnx.external_data_helper import load_external_data_for_model
from onnxsim import model_info, onnx_simplifier
from optimum.exporters.onnx import main_export


TARGET_IR = "dfg"
TARGET_NPU = "warboy-b0"

export_onnx = main_export


def simplify_onnx(input_model: Path, output_model: Path, overwrite_input_shapes: Dict[str, List[int]]) -> None:
    # https://github.com/daquexian/onnx-simplifier/blob/v0.4.28/onnxsim/onnx_simplifier.py#L479-L521
    print("Simplifying...")

    model = optimize_with_onnxruntime(input_model)

    model_opt, check_ok = onnx_simplifier.simplify(
        model,
        check_n=1,
        overwrite_input_shapes=overwrite_input_shapes,
        mutable_initializer=True,
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


def compile_onnx(input_model: Path, output_dfg: Path, output_dot: Path) -> None:
    model = onnx.load_model(input_model)
    try:
        graph = compile(model.SerializeToString(), target_ir=TARGET_IR, dot_graph=output_dot, target_npu=TARGET_NPU)
    except ValueError:
        model = move_initializer_to_input(model)
        graph = compile(model.SerializeToString(), target_ir=TARGET_IR, dot_graph=output_dot, target_npu=TARGET_NPU)

    with open(output_dfg, "wb") as f:
        f.write(bytes(graph))


def optimize_with_onnxruntime(input_model: Path) -> onnx.ModelProto:
    with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
        opt_onnx_path = file.name
    sess_options = ort.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # To enable model serialization after graph optimization
    sess_options.optimized_model_filepath = opt_onnx_path
    _ = ort.InferenceSession(input_model.as_posix(), sess_options, providers=["CPUExecutionProvider"])

    try:
        model = onnx.load_model(opt_onnx_path)
    except FileNotFoundError:
        model = onnx.load_model(opt_onnx_path, load_external_data=False)
        load_external_data_for_model(model, input_model.parent)

    return model


def move_initializer_to_input(model: onnx.ModelProto) -> onnx.ModelProto:
    # make every initializer graph_input, for compiler can't understand large(>2GB) onnx model.
    grpah_inputs = []
    for init in model.graph.initializer:
        np_array = onnx.numpy_helper.to_array(init)
        grpah_inputs.append(
            onnx.helper.make_tensor_value_info(
                init.name, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np_array.dtype], [*np_array.shape]
            )
        )
    model.graph.input.extend(grpah_inputs)
    model.graph.ClearField("initializer")
    return model
