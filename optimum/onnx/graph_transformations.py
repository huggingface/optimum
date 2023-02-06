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
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union

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
        graph_name (`str`):
            Name of the parent graph(graph of the control flow node).
        producer_name (`str`):
            Graph producer name.
        save_path (`str` or `Path`, *optional*):
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
    if not decoder_opset == decoder_with_past_opset:
        raise ValueError(
            f"Decoder's opset is {decoder_opset}, but decoder with past's opset is {decoder_with_past_opset}. Make sure having the same opset before merging."
        )

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

    if merged_model.ByteSize() < 2147483648:
        onnx.checker.check_model(merged_model)
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
        onnx.checker.check_model(save_path)
    else:
        logger.info("Merged ONNX model exceeds 2GB, the model will not be checked without `save_path` given.")

    return merged_model
