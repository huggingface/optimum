# https://github.com/daquexian/onnx-simplifier/blob/v0.3.10/onnxsim/onnx_simplifier.py
import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import onnx.helper  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnx.shape_inference  # type: ignore

import onnx  # type: ignore

Tensors = Dict[str, np.ndarray]
TensorShape = List[int]
TensorShapes = Dict[str, TensorShape]
TensorShapesWithOptionalKey = Dict[Optional[str], TensorShape]


def has_subgraph_in_node(node: onnx.NodeProto):
    for attr in node.attribute:
        if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
            return True
    return False


def get_all_subgraphs(model: onnx.ModelProto):
    graphs = [model.graph]
    for node in model.graph.node:
        if has_subgraph_in_node(node):
            for attr in node.attribute:
                graphs.append(attr.g)
    return graphs


def add_features_to_output(m: onnx.ModelProto, nodes: Sequence[onnx.NodeProto]) -> None:
    """
    Add features to output in pb, so that ONNX Runtime will output them.
    Note: the resulting model is not valid, because
    outputs of main graph should has other fields such as 'type'
    :param m: the model that will be run in ONNX Runtime
    :param nodes: nodes whose outputs will be added into the graph outputs
    """
    for node in nodes:
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)])


def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]


def get_value_info_all(m: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    for v in m.graph.value_info:
        if v.name == name:
            return v

    for v in m.graph.input:
        if v.name == name:
            return v

    for v in m.graph.output:
        if v.name == name:
            return v

    return None


def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
    """
    Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
    """
    v = get_value_info_all(m, name)
    if v is not None:
        return get_shape_from_value_info_proto(v)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))


def get_elem_type(m: onnx.ModelProto, name: str) -> int:
    v = get_value_info_all(m, name)
    if v is not None:
        return v.type.tensor_type.elem_type
    raise RuntimeError('Cannot get shape dtype "{}"'.format(name))


def get_np_type_from_elem_type(elem_type: int):
    sizes = (
        None,
        np.float32,
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.int32,
        np.int64,
        str,
        bool,
        np.float16,
        np.double,
        np.uint32,
        np.uint64,
        np.complex64,
        np.complex128,
        np.float16,
    )
    assert len(sizes) == 17
    size = sizes[elem_type]
    assert size is not None
    return size


def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = [ipt.name for ipt in get_inputs(model)]
    return input_names


def generate_specific_rand_input(
    model, input_shapes: TensorShapes, dynamic_input_shape: bool = False
):
    """
    Only generate rand inputs whose shape in `input_shapes`
    """

    for key, shape in input_shapes.items():
        shape_np = np.array(shape)
        if not np.all(shape_np > 0):
            # treat batch size as 1 automatically if dynamic_input_shape is True
            if (
                dynamic_input_shape and len(shape_np) >= 3 and np.all(shape_np[1:] > 0)
            ):  # fixed, not using config
                input_shapes[key] = [1] + shape[1:]
                continue

            raise RuntimeError(
                'The shape of input "{}" has dynamic size "{}", '
                'please try "--dynamic-input-shape" or determine '
                'the input size manually by "--input-shape xxx". '
                'Run "python3 -m onnxsim -h" for details'.format(key, shape)
            )

    inputs = {
        ipt: np.array(
            np.random.rand(*input_shapes[ipt]),
            dtype=get_np_type_from_elem_type(get_elem_type(model, ipt)),
        )
        for ipt in input_shapes
    }
    return inputs


def generate_all_rand_input(model, input_shapes: Optional[TensorShapes] = None):
    """
    Generate random array for all inputs of a model
    """
    if input_shapes is None:
        input_shapes = {}
    input_names = get_input_names(model)
    full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
    assert None not in input_shapes
    full_input_shapes.update(input_shapes)  # type: ignore
    return generate_specific_rand_input(model, full_input_shapes)


def is_non_deterministic_node(node: onnx.NodeProto) -> bool:
    # TODO: handle node with subgraph
    return node.op_type in [
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    ]


def is_non_deterministic_model(model: onnx.ModelProto) -> bool:
    return any([is_non_deterministic_node(node) for node in model.graph.node])


def get_constant_nodes(
    m: onnx.ModelProto, dynamic_input_shape: bool = False, include_subgraph: bool = False
) -> List[onnx.NodeProto]:
    const_nodes = []
    const_tensors = [x.name for x in m.graph.initializer]
    const_tensors.extend([node.output[0] for node in m.graph.node if node.op_type == "Constant"])

    # The output shape of some node types is determined by the input value
    # we consider the output of this node doesn't have constant shape,
    # so we do not simplify a such node even if the node is Shape op
    dynamic_tensors = []
    if dynamic_input_shape:
        dynamic_tensors.extend(get_input_names(m))

    def is_dynamic(node):
        if (
            node.op_type in ["NonMaxSuppression", "NonZero", "Unique"]
            and node.input[0] not in const_tensors
        ):
            return True
        if (
            node.op_type in ["Reshape", "Expand", "Upsample", "ConstantOfShape"]
            and len(node.input) > 1
            and node.input[1] not in const_tensors
        ):
            return True
        if node.op_type in ["Resize"] and (
            (len(node.input) > 2 and node.input[2] not in const_tensors)
            or (len(node.input) > 3 and node.input[3] not in const_tensors)
        ):
            return True
        return False

    def check_node(graph):
        for node in graph.node:
            if has_subgraph_in_node(node):
                # Skip this node if this node has subgraph in it
                # "If" node with const cond will be eliminated by onnxoptimizer
                if any(x in dynamic_tensors for x in node.input):
                    dynamic_tensors.extend(node.output)
                if include_subgraph:  # fixed not using config
                    for attr in node.attribute:
                        if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                            check_node(attr.g)
            elif any(x in dynamic_tensors for x in node.input):
                dynamic_tensors.extend(node.output)
            # Note "elif" here, only Shape op with non-dynamic input will be seen as const node
            elif node.op_type == "Shape":
                const_nodes.append(node)
                const_tensors.extend(node.output)
            elif is_dynamic(node):
                dynamic_tensors.extend(node.output)
            elif node.op_type in ["DequantizeLinear", "QuantizeLinear"]:
                # Skip QuantizeLinear and DequantizeLinear to preserve quantization info
                pass
            elif all([x in const_tensors for x in node.input]) and not is_non_deterministic_node(
                node
            ):
                # Skip these nodes to avoid bloating the model size
                if node.op_type in ["Tile"]:
                    continue
                const_nodes.append(node)
                const_tensors.extend(node.output)

    check_node(m.graph)
    return copy.deepcopy(const_nodes)


def eliminate_const_nodes(
    model: onnx.ModelProto, const_nodes: Sequence[onnx.NodeProto], res: Tensors
) -> onnx.ModelProto:
    """
    :param model: the original onnx model
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: the simplified onnx model. Redundant ops are all removed.
    """

    def recursive_eliminate_const_nodes_in_graph(graph, const_nodes, res):
        new_nodes = []
        for i, node in enumerate(graph.node):
            if node in const_nodes:
                for output in node.output:
                    new_node = copy.deepcopy(node)
                    new_node.name = "node_" + output
                    new_node.op_type = "Constant"
                    new_attr = onnx.helper.make_attribute(
                        "value", onnx.numpy_helper.from_array(res[output], name=output)
                    )
                    del new_node.input[:]
                    del new_node.attribute[:]
                    del new_node.output[:]
                    new_node.output.extend([output])
                    new_node.attribute.extend([new_attr])
                    new_nodes.append(new_node)
            else:
                new_nodes.append(node)
                if has_subgraph_in_node(node):
                    for attr in node.attribute:
                        if attr.g is None:
                            continue
                        recursive_eliminate_const_nodes_in_graph(attr.g, const_nodes, res)
        del graph.node[:]
        graph.node.extend(new_nodes)

    recursive_eliminate_const_nodes_in_graph(model.graph, const_nodes, res)

    return model


def clean_constant_nodes(const_nodes: Sequence[onnx.NodeProto], res: Tensors):
    """
    It seems not needed since commit 6f2a72, but maybe it still prevents some unknown bug
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: The constant nodes which have an output in res
    """
    return [node for node in const_nodes if node.output[0] in res]


def check_and_update_input_shapes(
    model: onnx.ModelProto,
    input_shapes: TensorShapesWithOptionalKey,
    dynamic_input_shape: bool = False,
) -> TensorShapes:
    input_names = get_input_names(model)
    if None in input_shapes:
        if len(input_names) == 1:
            input_shapes[input_names[0]] = input_shapes[None]
            del input_shapes[None]
        else:
            raise RuntimeError(
                'The model has more than 1 inputs, please use the format "input_name:dim0,dim1,...,dimN" in --input-shape'
            )
    for x in input_shapes:
        if x not in input_names:
            raise RuntimeError('The model doesn\'t have input named "{}"'.format(x))

    # Overwrite model input shape
    if not dynamic_input_shape:
        for name, input_shape in input_shapes.items():
            for ipt in model.graph.input:
                if ipt.name == name:
                    for i, dim in enumerate(ipt.type.tensor_type.shape.dim):
                        dim.dim_value = input_shape[i]

    return input_shapes  # type: ignore


# https://github.com/daquexian/onnx-simplifier/blob/v0.4.33/onnxsim/model_info.py
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

from rich import print
from rich.table import Table
from rich.text import Text

import onnx

__all__ = ["ModelInfo", "print_simplifying_info"]


def human_readable_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


class ModelInfo:
    """
    Model info contains:
    1. Num of every op
    2. Model size
    TODO:
    Based on onnx runtime, get
    1、FLOPs
    2、forward memory footprint
    3、memory access
    4、compute density
    """

    def get_info(self, graph: onnx.GraphProto) -> Tuple[Dict[str, int], int]:
        op_nums = defaultdict(int)
        model_size = 0
        for node in graph.node:
            op_nums[node.op_type] += 1
            for attr in node.attribute:
                sub_graphs = []
                if attr.g is not None:
                    sub_graphs.append(attr.g)
                if attr.graphs is not None:
                    sub_graphs.extend(attr.graphs)
                for sub_graph in sub_graphs:
                    sub_op_nums, sub_model_size = self.get_info(sub_graph)
                    op_nums = defaultdict(
                        int,
                        {k: op_nums[k] + sub_op_nums[k] for k in set(op_nums) | set(sub_op_nums)},
                    )
                    model_size += sub_model_size
        op_nums["Constant"] += len(graph.initializer)
        model_size += graph.ByteSize()
        return op_nums, model_size

    def __init__(self, model: onnx.ModelProto):
        self.op_nums, self.model_size = self.get_info(model.graph)


def print_simplifying_info(model_ori: onnx.ModelProto, model_opt: onnx.ModelProto) -> None:
    """
    --------------------------------------------------------
    |             | original model | simplified model |
    --------------------------------------------------------
    | ****        | ****           | ****             |
    --------------------------------------------------------
    | Model Size  | ****           | ****             |
    --------------------------------------------------------
    """
    ori_info = ModelInfo(model_ori)
    opt_info = ModelInfo(model_opt)
    table = Table()
    table.add_column("")
    table.add_column("Original Model")
    table.add_column("Simplified Model")

    def add_row(
        table: Table,
        key,
        ori_data,
        opt_data,
        is_better: Callable[[Any, Any], Any],
        postprocess: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if postprocess is None:
            postprocess = str
        if is_better(opt_data, ori_data):
            table.add_row(
                key, postprocess(ori_data), Text(postprocess(opt_data), style="bold green1")
            )
        else:
            table.add_row(key, postprocess(ori_data), postprocess(opt_data))

    for key in sorted(list(set(ori_info.op_nums.keys()) | set(opt_info.op_nums.keys()))):
        add_row(
            table, key, ori_info.op_nums[key], opt_info.op_nums[key], lambda opt, ori: opt < ori
        )
    add_row(
        table,
        "Model Size",
        ori_info.model_size,
        opt_info.model_size,
        lambda opt, ori: opt < ori,
        postprocess=human_readable_size,
    )
    print(table)
