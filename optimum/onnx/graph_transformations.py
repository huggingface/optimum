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

import copy

import onnx
from onnx import ModelProto

from ..utils import logging


logger = logging.get_logger()


from .transformations_utils import (
    _create_name_sharing_dict,
    _deduplicated_cross_model_initializers,
    _find_duplicate_weights,
    _get_all_inputs,
    _get_onnx_opset,
    _remove_redundant_initializers,
    _replace_input_names,
    _unify_onnx_outputs,
    cast_int64_tensorproto_to_int32,
)


def remove_duplicate_weights(model: ModelProto, inplace: bool = False) -> ModelProto:
    """
    Finds and removes duplicate weights in a model by keeping only unique weights, and make the duplicate values point
    to them.

    Args:
        model (`onnx.ModelProto`): The model to remove duplicates from.
        inplace (`bool`, defaults to False): Whether to perform this transformation inplace.

    Returns:
        `onnx.ModelProto`: The model without duplicates.
    """
    if not inplace:
        model = copy.deepcopy(model)
    duplicates = _find_duplicate_weights(model)
    name_sharing_dict = _create_name_sharing_dict(duplicates)

    _replace_input_names(model, name_sharing_dict)
    _remove_redundant_initializers(model, name_sharing_dict)

    return model


def replace_atenops_to_gather(model: ModelProto) -> ModelProto:
    """
    Replaces broken ATenOp nodes back to Gather nodes.

    Args:
        model (`onnx.ModelProto`):
            The ONNX model to fix.

    Returns:
        `onnx.ModelProto`: The ONNX model fixed.
    """
    nodes = model.graph.node

    for node in nodes:
        if node.op_type in ["ATenOp", "ATen"]:
            op_num = node.name.split("_")[-1]
            new_node = onnx.helper.make_node(
                "Gather",
                name="Gather_" + op_num,
                inputs=[node.input[0], node.input[1]],
                outputs=node.output,
            )

            model.graph.node.remove(node)
            model.graph.node.insert(int(op_num), new_node)

    onnx.checker.check_model(model)
    return model


def merge_decoders(
    decoder: ModelProto,
    decoder_with_past: ModelProto,
    graph_name: str = "merged",
    producer_name: str = "optimum-onnx",
) -> ModelProto:
    """
    Fuses decoder ONNX model and decoder with past ONNX model into one ONNX model with if logic.

    Args:
        decoder (`onnx.ModelProto`): Decoder ONNX model.
        decoder_with_past (`onnx.ModelProto`): Decoder with past ONNX model.
        graph_name (`str`): Name of the parent graph(graph of the control flow node).
        producer_name (`str`): Graph producer name.

    Returns:
        `~onnx.ModelProto`: The fused decoder ONNX model.
    """

    _unify_onnx_outputs(decoder, decoder_with_past)
    all_inputs = _get_all_inputs([decoder, decoder_with_past])
    deduplicated_initializers = _deduplicated_cross_model_initializers([decoder, decoder_with_past], suffix=graph_name)

    # Make subgraphs
    no_past_branch = onnx.helper.make_graph(
        nodes=decoder.graph.node,
        name="no_past",
        inputs=[],
        outputs=decoder.graph.output,
        initializer=[],
    )
    with_past_branch = onnx.helper.make_graph(
        nodes=decoder_with_past.graph.node,
        name="with_past",
        inputs=[],
        outputs=decoder_with_past.graph.output,
        initializer=[],
    )

    # Merge subgraphs with a `If` node
    use_cache = onnx.helper.make_tensor_value_info(
        name="use_cache",
        elem_type=onnx.TensorProto.BOOL,
        shape=[1],
    )
    if_node = onnx.helper.make_node(
        "If",
        inputs=["use_cache"],
        outputs=[output.name for output in no_past_branch.output],
        name="optimum::if",
        then_branch=with_past_branch,
        else_branch=no_past_branch,
    )
    merged_graph = onnx.helper.make_graph(
        nodes=[if_node],
        name=graph_name,
        inputs=all_inputs + [use_cache],
        outputs=no_past_branch.output,
        initializer=deduplicated_initializers,
    )
    decoder_opset = _get_onnx_opset(decoder)
    decoder_with_past_opset = _get_onnx_opset(decoder_with_past)
    if not decoder_opset == decoder_with_past_opset:
        raise ValueError(
            f"Decoder's opset is {decoder_opset}, but decoder with past's opset is {decoder_with_past_opset}. Make sure having the same opset before merging."
        )
    merged_model = onnx.helper.make_model(
        merged_graph,
        producer_name=producer_name,
        opset_imports=[
            onnx.helper.make_opsetid(
                domain=onnx.defs.ONNX_DOMAIN,
                version=decoder_opset,
            )
        ],
    )
    onnx.checker.check_model(merged_model)

    return merged_model


def model_to_int32(model: ModelProto) -> ModelProto:
    """
    Convert node inputs of `Slice` nodes from int64 to int32, casting the out of range values.

    The constant node inputs are stored in `model.graph.node`, and the sole way to check which node
    they are consumed by is to iterate over nodes and check `node.input` for a match.

    Note that constant inputs to nodes as `Squeeze`, `Unsqueeze` can not be converted to int32, as the
    these operators explicitely expect int64 inputs according to ONNX specifications:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md
    """
    map_input_node = {}

    for node in model.graph.node:
        for input_name in node.input:
            map_input_node[input_name] = {"op_type": node.op_type}

    for node in model.graph.node:
        if (
            node.op_type == "Constant"
            and node.attribute[0].t.data_type == 7  # int64
            and f"{node.name}_output_0" in map_input_node
            and map_input_node[node.name + "_output_0"]["op_type"] == "Slice"
        ):
            logger.debug(f"Converting {node.name} to int32")
            cast_int64_tensorproto_to_int32(node.attribute[0].t)

    return model
