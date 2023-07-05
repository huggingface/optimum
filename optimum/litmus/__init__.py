import copy
from pathlib import Path
from typing import Dict, List, Union

from furiosa.tools.compiler.api import compile
import onnx
from optimum.exporters.onnx import main_export
from optimum.litmus import onnxsim, utils

TARGET_IR = "dfg"
TARGET_NPU = "warboy-b0"

export_onnx = main_export


def simplify_onnx(
    input_model: Union[Path, onnx.ModelProto],
    output_model: Path,
    overwrite_input_shapes: Dict[str, List[int]],
) -> onnx.ModelProto:
    model_orig = utils.load_onnx(input_model) if isinstance(input_model, Path) else input_model
    model_copy = copy.deepcopy(model_orig)

    model_opt = utils.optimize_with_onnxruntime(model_copy)
    model_opt = utils.constant_folding(
        model_opt,
        overwrite_input_shapes=onnxsim.check_and_update_input_shapes(
            model_opt, overwrite_input_shapes
        ),
    )
    model_opt = utils.optimize_with_onnxruntime(model_opt)
    model_opt = utils.infer_onnx_tensor_shapes(model_opt)

    utils.save_onnx(model_opt, output_model)

    utils.check_opt_model(
        model_opt, model_orig, n_times=5, input_shapes=overwrite_input_shapes
    )
    onnxsim.print_simplifying_info(model_orig, model_opt)
    return model_opt


def compile_onnx(
    input_model: Union[Path, onnx.ModelProto], output_dfg: Path, output_dot: Path
) -> None:
    if isinstance(input_model, Path):
        input_model = utils.load_onnx(input_model)

    model = input_model

    if model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
        # make every initializer graph_input, for compiler can't understand large(>2GB) onnx model.
        model = utils.move_initializer_to_input(model)

    graph = compile(
        model.SerializeToString(),
        target_ir=TARGET_IR,
        dot_graph=output_dot,
        target_npu=TARGET_NPU,
    )

    with open(output_dfg, "wb") as f:
        f.write(bytes(graph))
