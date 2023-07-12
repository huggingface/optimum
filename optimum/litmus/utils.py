import argparse
from collections import OrderedDict
import copy
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

import onnx
from optimum.litmus import onnxsim


def constant_folding(
    model: onnx.ModelProto, overwrite_input_shapes: Dict[str, List[int]]
) -> onnx.ModelProto:
    const_nodes = onnxsim.get_constant_nodes(model, dynamic_input_shape=False)
    res = forward_onnx_for_node_outputs(
        model, const_nodes, input_shapes=overwrite_input_shapes, input_data={}, custom_lib=None
    )
    const_nodes = onnxsim.clean_constant_nodes(const_nodes, res)
    model = onnxsim.eliminate_const_nodes(model, const_nodes, res)
    check_onnx(model)
    return model


def forward_onnx_for_node_outputs(
    model: onnx.ModelProto,
    nodes: Sequence[onnx.NodeProto],
    input_shapes: Optional[onnxsim.TensorShapes] = None,
    input_data: Optional[onnxsim.Tensors] = None,
    custom_lib: Optional[str] = None,
    include_subgraph: bool = False,
) -> onnxsim.Tensors:
    if input_shapes is None:
        input_shapes = {}
    model = copy.deepcopy(model)

    onnxsim.add_features_to_output(model, nodes)
    output_names = []
    for node in nodes:
        output_names.extend(node.output)

    if include_subgraph:  # fixed: not using config
        subgraphs = onnxsim.get_all_subgraphs(model)
        for i in range(1, len(subgraphs)):
            subgraphs[0].node.extend(subgraphs[i].node)
        model = onnx.utils.Extractor(model).extract_model([], output_names)
        check_onnx(model)
    res = forward_onnx(
        model,
        input_data=input_data,
        input_shapes=input_shapes,
        outputs=output_names,
        custom_lib=custom_lib,
    )
    return res


# https://github.com/daquexian/onnx-simplifier/blob/v0.4.33/onnxsim/model_checking.py#L15-L185
def check_opt_model(
    model_opt: Union[str, onnx.ModelProto],
    model_ori: Union[str, onnx.ModelProto],
    n_times: int = 5,
    input_shapes: Optional[onnxsim.TensorShapes] = None,
    input_data: Optional[onnxsim.Tensors] = None,
    custom_lib: Optional[str] = None,
    verbose=True,
) -> bool:
    """
    :param model_opt: The simplified ONNX model
    :param model_ori: The original ONNX model
    :param n_times: Generate n random inputs
    :param input_shapes: Shapes of generated random inputs
    :param input_data: User-given data instead of random generated data
    :param custom_lib: ONNX Runtime custom lib for custom ops
    """

    if input_shapes is None:
        input_shapes = {}

    check_onnx(model_opt)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 3
    sess_opt = initiate_onnxruntime_session(model_opt, sess_options)
    sess_ori = initiate_onnxruntime_session(model_ori, sess_options)

    assert not [
        node for node in model_opt.graph.node if node.op_type == "Shape"
    ], "Shape operator remains in simplified ONNX graph. It should be further simplified."

    for i in range(n_times):
        print(f"Checking {i+1}/{n_times}...")
        if input_data is None:
            inputs = onnxsim.generate_all_rand_input(model_opt, input_shapes=input_shapes)
        else:
            inputs = input_data
        res_ori = forward_onnx(model_ori, inputs, custom_lib, sess=sess_opt)
        res_opt = forward_onnx(model_opt, inputs, custom_lib, sess=sess_ori)

        for name in res_opt.keys():
            if not np.allclose(res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5):
                if verbose:
                    print(
                        "Tensor {} changes after optimization. The max diff is {}.".format(
                            name, np.max(np.abs(res_opt[name] - res_ori[name]))
                        )
                    )
                    print("After optimization:")
                    print(res_opt[name])
                    print("Before optimization:")
                    print(res_ori[name])
                    print("----------------")
                return False
    return True


def load_onnx(input_model: Union[Path, str]) -> onnx.ModelProto:
    input_model = Path(input_model) if isinstance(input_model, str) else input_model
    onnx_model = onnx.load_model(input_model.as_posix())
    return onnx_model


def save_onnx(model: onnx.ModelProto, output_model: Union[Path, str]) -> None:
    model = copy.deepcopy(model)
    if isinstance(output_model, str):
        output_model = Path(output_model)

    if model.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
        onnx.save(model, output_model.as_posix())
    else:
        external_data_path = output_model.parent / (output_model.name + "_data")
        if external_data_path.exists():
            external_data_path.unlink()

        # large models
        onnx.save(
            model,
            output_model.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(output_model) + "_data",
        )


def infer_onnx_tensor_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    if model.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
        model = onnx.shape_inference.infer_shapes(model)
    else:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
            inferred_model_path = file.name
        save_onnx(model, inferred_model_path)
        onnx.shape_inference.infer_shapes_path(inferred_model_path)
        model = load_onnx(inferred_model_path)
    return model


def make_onnx_model(
    inputs: List[onnx.ValueInfoProto],
    nodes: List[onnx.NodeProto],
    initializers: List[onnx.TensorProto],
    outputs: List[onnx.ValueInfoProto],
    opset_imports: List[onnx.OperatorSetIdProto],
    graph_name: str,
    producer_name: Optional[str] = None,
) -> onnx.ModelProto:
    graph = onnx.helper.make_graph(
        nodes=nodes, inputs=inputs, outputs=outputs, initializer=initializers, name=graph_name
    )
    model = onnx.helper.make_model(graph, opset_imports=opset_imports, producer_name=producer_name)
    check_onnx(model)
    return model


def check_onnx(model: onnx.ModelProto) -> None:
    model = copy.deepcopy(model)
    if model.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
        onnx.checker.check_model(model, full_check=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
            tmp_onnx_path = file.name
        save_onnx(model, tmp_onnx_path)
        onnx.checker.check_model(tmp_onnx_path, full_check=True)


def forward_onnx(
    model: onnx.ModelProto,
    input_data: Optional[onnxsim.Tensors] = None,
    input_shapes: Optional[onnxsim.TensorShapes] = None,
    outputs: Optional[Sequence[str]] = None,
    custom_lib: Optional[str] = None,
    sess: Optional[ort.InferenceSession] = None,
) -> onnxsim.Tensors:
    if outputs is not None and len(outputs) == 0:
        return {}
    if input_shapes is None:
        input_shapes = {}

    if sess is None:
        sess_options = ort.SessionOptions()
        if custom_lib is not None:
            if os.path.exists(custom_lib):
                sess_options.register_custom_ops_library(custom_lib)
            else:
                print("No such file '{}'".format(custom_lib), file=sys.stderr)
                exit(1)

        sess_options.graph_optimization_level = ort.GraphOptimizationLevel(0)
        sess_options.log_severity_level = 3
        sess = initiate_onnxruntime_session(model, sess_options)

    input_names = onnxsim.get_input_names(model)
    inputs = {}
    for name in input_names:
        if input_data is not None and input_data.get(name, None) is not None:
            inputs[name] = input_data[name]
        else:
            if input_shapes is not None and input_shapes.get(name, None) is not None:
                shape = input_shapes[name]
            else:
                shape = onnxsim.get_shape(model, name)
            inputs.update(onnxsim.generate_specific_rand_input(model, {name: shape}))

    if not outputs:
        outputs = [x.name for x in sess.get_outputs()]
    run_options = ort.RunOptions()
    run_options.log_severity_level = 3
    res = OrderedDict(zip(outputs, sess.run(outputs, inputs, run_options=run_options)))
    return res


def move_initializer_to_input(model: onnx.ModelProto) -> onnx.ModelProto:
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


def get_subgraphs(model: onnx.ModelProto) -> onnx.AttributeProto.GRAPHS:
    subgraphs = []
    for node in model.graph.node:
        if onnxsim.has_subgraph_in_node(node):
            for attr in node.attribute:
                subgraphs.append(attr.g)
    return subgraphs


def make_dynamic_axis_fixed(value_info: onnx.ValueInfoProto, dim_param, dim_value) -> None:
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param == dim_param:
            dim.dim_value = dim_value


def has_dynamic_axis(value_info: onnx.ValueInfoProto) -> bool:
    return any(dim.dim_param for dim in value_info.type.tensor_type.shape.dim)


def initiate_onnxruntime_session(
    model: onnx.ModelProto, sess_options: ort.SessionOptions
) -> ort.InferenceSession:
    model = copy.deepcopy(model)
    if model.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
        sess = ort.InferenceSession(
            model.SerializeToString(), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
            tmp_onnx_path = file.name
        save_onnx(model, tmp_onnx_path)
        sess = ort.InferenceSession(
            tmp_onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    return sess


def optimize_with_onnxruntime(model: onnx.ModelProto) -> onnx.ModelProto:
    with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
        opt_onnx_path = file.name
    save_onnx(model, opt_onnx_path)

    sess_options = ort.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.log_severity_level = 3
    # To enable model serialization after graph optimization
    sess_options.optimized_model_filepath = opt_onnx_path
    _ = ort.InferenceSession(
        opt_onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    model_opt = load_onnx(opt_onnx_path)
    return model_opt


def check_non_negative(value) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid negative int value" % value)
    return ivalue
