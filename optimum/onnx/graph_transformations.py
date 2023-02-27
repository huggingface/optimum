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
import os
from pathlib import Path
from typing import Optional, Union

import onnx
from onnx import ModelProto

from ..utils import logging
from .transformations_utils import (
    _create_name_sharing_dict,
    _deduplicated_cross_model_initializers,
    _find_duplicate_initializers,
    _get_all_inputs,
    _get_onnx_opset,
    _remove_redundant_initializers,
    _replace_input_names,
    _unify_onnx_outputs,
    cast_int64_tensorproto_to_int32,
)


logger = logging.get_logger()


ONNX_BYTE_SIZE_LIMIT = 2147483648


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
    duplicates = _find_duplicate_initializers(models=[model])
    name_sharing_dict = _create_name_sharing_dict(duplicates)

    _replace_input_names(models=[model], name_sharing_dict=name_sharing_dict)
    _remove_redundant_initializers(models=[model], name_sharing_dict=name_sharing_dict)

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
    decoder: Union[ModelProto, Path, str],
    decoder_with_past: Union[ModelProto, Path, str],
    graph_name: str = "merged",
    producer_name: str = "optimum-onnx",
    save_path: Optional[Union[str, Path]] = None,
) -> ModelProto:
    """
    Fuses decoder ONNX model and decoder with past ONNX model into one ONNX model with if logic.

    Args:
        decoder (`Union[ModelProto, Path, str]`):
            Decoder ONNX model.
        decoder_with_past (`Union[ModelProto, Path, str]`):
            Decoder with past ONNX model.
        graph_name (`str`, defaults to `"merged"`):
            Name of the parent graph (graph of the control flow node).
        producer_name (`str`, defaults to `"optimum-onnx"`):
            Graph producer name.
        save_path (`Optional[Union[str, Path]]`, defaults to `None`):
            The path to save merged ONNX model. The model will be saved if the path is given.

    Returns:
        `~onnx.ModelProto`: The fused decoder ONNX model.
    """
    if isinstance(decoder, (str, Path)):
        decoder = Path(decoder).as_posix()
        decoder = onnx.load(decoder)

    if isinstance(decoder_with_past, (str, Path)):
        decoder_with_past = Path(decoder_with_past).as_posix()
        decoder_with_past = onnx.load(decoder_with_past)

    decoder_opset = _get_onnx_opset(decoder)
    decoder_with_past_opset = _get_onnx_opset(decoder_with_past)
    if decoder_opset != decoder_with_past_opset:
        raise ValueError(
            f"Decoder's opset is {decoder_opset}, but decoder with past's opset is {decoder_with_past_opset}. Make sure having the same opset before merging."
        )

    _unify_onnx_outputs(decoder, decoder_with_past)
    all_inputs = _get_all_inputs([decoder, decoder_with_past])

    # Replace the axis name `sequence_length` of the attention_mask input by `attention_mask_sequence_length`.
    # This is because the merged model `input_ids` and `attention_mask` inputs may not always have the same length on the 2nd axis.
    # In the first pass, `input_ids` and `attention_mask` are indeed of the same length, but in later pass `input_ids` is of length 1
    # while `attention_mask` is of length `past_sequence_length + 1`
    for _, inp in enumerate(all_inputs):
        if inp.name == "attention_mask":
            if inp.type.tensor_type.shape.dim[1].dim_param != "sequence_length":
                raise ValueError("Expected attention_mask second axis to be dynamic and named `sequence_length`.")
            inp.type.tensor_type.shape.dim[1].dim_param = "attention_mask_sequence_length"

    deduplicated_initializers = _deduplicated_cross_model_initializers([decoder, decoder_with_past], suffix=graph_name)

    # Keep initializers of dim 0 (or dim 1 + int32/int64) in subgraphs for readability purposes, and also because
    # ONNX Runtime breaks after optimization + merge if they are not
    decoder_initializers = []
    for initializer in decoder.graph.initializer:
        if len(initializer.dims) == 0 or (len(initializer.dims) == 1 and initializer.data_type in [6, 7]):
            decoder_initializers.append(initializer)

    decoder_with_past_initializers = []
    for initializer in decoder_with_past.graph.initializer:
        if len(initializer.dims) == 0 or (len(initializer.dims) == 1 and initializer.data_type in [6, 7]):
            decoder_with_past_initializers.append(initializer)

    # Make subgraphs
    no_past_branch = onnx.helper.make_graph(
        nodes=decoder.graph.node,
        name="no_past",
        inputs=[],
        outputs=decoder.graph.output,
        initializer=decoder_initializers,
    )
    with_past_branch = onnx.helper.make_graph(
        nodes=decoder_with_past.graph.node,
        name="with_past",
        inputs=[],
        outputs=decoder_with_past.graph.output,
        initializer=decoder_with_past_initializers,
    )

    # Merge subgraphs with a `If` node
    use_cache_branch = onnx.helper.make_tensor_value_info(
        name="use_cache_branch",
        elem_type=onnx.TensorProto.BOOL,
        shape=[1],
    )
    if_node = onnx.helper.make_node(
        "If",
        inputs=["use_cache_branch"],
        outputs=[output.name for output in no_past_branch.output],
        name="optimum::if",
        then_branch=with_past_branch,
        else_branch=no_past_branch,
    )
    merged_graph = onnx.helper.make_graph(
        nodes=[if_node],
        name=graph_name,
        inputs=all_inputs + [use_cache_branch],
        outputs=no_past_branch.output,
        initializer=deduplicated_initializers,
    )

    # Preserve imports from the decoder without/with past ONNX
    opset_imports = []
    opset_domains = set()
    for opset_import in list(decoder.opset_import) + list(decoder_with_past.opset_import):
        if opset_import.domain not in opset_domains:
            opset_imports.append(opset_import)
            opset_domains.add(opset_import.domain)

    merged_model = onnx.helper.make_model(merged_graph, producer_name=producer_name, opset_imports=opset_imports)

    # For the try catch, refer to https://github.com/microsoft/onnxruntime/issues/14768
    if merged_model.ByteSize() < ONNX_BYTE_SIZE_LIMIT:
        try:
            onnx.checker.check_model(merged_model)
        except Exception as e:
            if "No Op registered for" in str(e):
                pass
            else:
                raise e
        if save_path:
            save_path = Path(save_path).as_posix()
            onnx.save(merged_model, save_path)
    elif save_path is not None:
        save_path = Path(save_path).as_posix()
        onnx.save(
            merged_model,
            save_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(save_path) + "_data",
        )
        try:
            onnx.checker.check_model(merged_model)
        except Exception as e:
            if "No Op registered for" in str(e):
                pass
            else:
                raise e
    else:
        logger.info("Merged ONNX model exceeds 2GB, the model will not be checked without `save_path` given.")

    return merged_model


def cast_slice_nodes_inputs_to_int32(model: ModelProto) -> ModelProto:
    """
    Convert node inputs of `Slice` nodes from int64 to int32, casting the out of range values.

    The constant node inputs are stored in `model.graph.node`, and the sole way to check which node
    they are consumed by is to iterate over nodes and check `node.input` for a match.

    Note that constant inputs to nodes as `Squeeze`, `Unsqueeze` can not be converted to int32, as the
    these operators explicitely expect int64 inputs according to ONNX specifications:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md
    """
    map_input_node = {}
    map_node_inputs = {}

    for node in model.graph.node:
        for input_name in node.input:
            map_input_node[input_name] = {"op_type": node.op_type, "node_name": node.name}
        map_node_inputs[node.name] = node.input

    for node in model.graph.node:
        if (
            node.op_type == "Constant"
            and node.attribute[0].t.data_type == 7  # int64
            and f"{node.name}_output_0" in map_input_node
            and map_input_node[node.name + "_output_0"]["op_type"] == "Slice"
        ):
            logger.debug(f"Converting {node.name} to int32")

            # `Slice` node is homogeneous (requires parameters of same type), hence cast to int32 only if all of its inputs are constants
            # refer to onnx/defs/schema.h
            cast = all(
                "Constant" in inp for inp in map_node_inputs[map_input_node[node.name + "_output_0"]["node_name"]][1:]
            )
            cast_int64_tensorproto_to_int32(node.attribute[0].t, cast=cast)

    return model
