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
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import onnx
from onnx import ModelProto, ValueInfoProto
from onnxsim import simplify
from onnxsim.model_info import print_simplifying_info


def _find_duplicate_weights(model) -> DefaultDict[Tuple[int, bytes], Set[str]]:
    return _find_duplicate_initializers(model.graph.initializer)


def _find_duplicate_initializers(initializers) -> DefaultDict[Tuple[int, bytes], Set[str]]:
    duplicates = defaultdict(set)
    for initializer in initializers:
        for data_attr in ["raw_data", "int32_data", "int64_data", "uint64_data", "float_data", "double_data"]:
            tensor_data = getattr(initializer, data_attr)
            if tensor_data:
                tensor_data = tuple(tensor_data)
                break
        duplicates[(initializer.data_type, tensor_data)].add(initializer.name)
    return duplicates


def _create_name_sharing_dict(
    duplicate_weights: DefaultDict[Tuple[int, bytes], Set[str]], suffix: str = None
) -> Dict[str, str]:
    def _create_name_sharing_dict_for_duplicates(duplicates: Set[str]) -> Dict[str, str]:
        common_name = duplicates.pop()
        duplicates.add(common_name)
        if suffix:
            return {k: common_name + "_" + suffix for k in duplicates}
        else:
            return {k: common_name for k in duplicates}

    name_sharing_dict = {}
    for duplicates in duplicate_weights.values():
        name_sharing_dict.update(_create_name_sharing_dict_for_duplicates(duplicates))
    return name_sharing_dict


def _replace_input_names(model: ModelProto, name_sharing_dict: Dict[str, str]):
    for node in model.graph.node:
        for i in range(len(node.input)):
            node.input[i] = name_sharing_dict.get(node.input[i], node.input[i])


def _remove_redundant_initializers(model: ModelProto, name_sharing_dict: Dict[str, str]):
    to_pop = []
    for idx, initializer in enumerate(model.graph.initializer):
        if initializer.name != name_sharing_dict[initializer.name]:
            to_pop.append(idx)

    for idx in sorted(to_pop, reverse=True):
        model.graph.initializer.pop(idx)


def remove_duplicate_weights(model: ModelProto, inplace: bool = False) -> ModelProto:
    """
    Finds and removes duplicate weights in a model by keeping only unique weights, and make the duplicate values point
    to them.

    Args:
        model (`~onnx.ModelProto`): The model to remove duplicates from.
        inplace (`bool`, defaults to False): Whether to perform this transformation inplace.

    Returns:
        `~onnx.ModelProto`: The model without duplicates.
    """
    if not inplace:
        model = copy.deepcopy(model)
    duplicates = _find_duplicate_weights(model)
    name_sharing_dict = _create_name_sharing_dict(duplicates)

    _replace_input_names(model, name_sharing_dict)
    _remove_redundant_initializers(model, name_sharing_dict)

    return model


def replace_atenops_to_gather(model: ModelProto):
    """
    Replaces broken ATenOp nodes back to Gather nodes.

    Args:
        model (`~onnx.ModelProto`): The ONNX model to fix.
        inplace (`bool`, defaults to False): Whether to perform this transformation inplace.

    Returns:
        `~onnx.ModelProto`: The ONNX model fixed.
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


def _infer_output_shape(output: ValueInfoProto):
    output_shape = []
    for dim in output.type.tensor_type.shape.dim:
        if getattr(dim, "dim_param"):
            output_shape.append(getattr(dim, "dim_param"))
        elif getattr(dim, "dim_value"):
            output_shape.append(getattr(dim, "dim_value"))
        else:
            raise ValueError(f"Can not find `dim_param` nor `dim_value` in the output dimension info.")

    return output_shape


def _check_num_outputs(model1: ModelProto, model2: ModelProto):
    if not len(model1.graph.output) == len(model2.graph.output):
        raise ValueError(
            f"Two model protos need to have same outputs. But one has {len(model1.graph.output)} "
            f"outputs while the other has {len(model2.graph.output)} outputs."
        )


def _unify_onnx_outputs(model1: ModelProto, model2: ModelProto):
    """
    Unify the outputs of deux ONNX model protos. The outputs of model1 will be replaced by outputs of model2.
    """
    _check_num_outputs(model1, model2)

    for idx in range(len(model1.graph.output)):
        model_output_1 = model1.graph.output[idx]
        model_output_2 = model2.graph.output[idx]
        if not model_output_1 == model_output_2:
            if not (
                model_output_1.name == model_output_2.name
                and model_output_1.type.tensor_type.elem_type == model_output_2.type.tensor_type.elem_type
            ):
                raise ValueError(
                    f"Can not match {model_output_1.name} with {model_output_2.name}. Make sure your"
                    f" model protos have same outputs, have same data types and are in the same order."
                )
            model1.graph.output.remove(model_output_1)

            new_output = onnx.helper.make_tensor_value_info(
                model_output_2.name,
                model_output_2.type.tensor_type.elem_type,
                _infer_output_shape(model_output_2),
            )
            model1.graph.output.insert(idx, new_output)

    if not all(
        model_output_1 == model_output_2
        for model_output_1, model_output_2 in zip(model1.graph.output, model2.graph.output)
    ):
        raise RuntimeError(f"Failed to unify outputs of given ONNX model protos.")


def _get_all_inputs(model_list: List[ModelProto]):
    inputs = []
    input_names = []
    for model in model_list:
        for input in model.graph.input:
            if input.name not in input_names:
                input_names.append(input.name)
                inputs.append(input)
    return inputs


def _get_onnx_opset(model: ModelProto):
    opset_import = model.opset_import[0]
    return getattr(opset_import, "version")


def simplify_onnx_graph(model: ModelProto, print_info: bool = True) -> ModelProto:

    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated."
    if print_info:
        print_simplifying_info(model, model_simp)

    return model_simp


def _deduplicated_cross_model_initializers(models: List[ModelProto], suffix: str = None):

    all_initializers = list()
    for model in models:
        all_initializers += list(model.graph.initializer)

    duplicates = _find_duplicate_initializers(all_initializers)
    name_sharing_dict = _create_name_sharing_dict(duplicates, suffix=suffix)
    for model in models:
        _replace_input_names(model, name_sharing_dict)

    deduplicated_initializers = list()
    deduplicated_name = list()

    for initializer in all_initializers:
        if name_sharing_dict[initializer.name] not in deduplicated_name:
            deduplicated_name.append(name_sharing_dict[initializer.name])
            initializer.name = name_sharing_dict[initializer.name]
            deduplicated_initializers.append(initializer)

    return deduplicated_initializers


def merge_decoders(
    decoder: ModelProto,
    decoder_with_past: ModelProto,
    graph_name: str = "merged",
    producer_name: str = "optimum-onnx",
    simplify_graph: bool = False,
) -> ModelProto:
    """
    Fuses decoder ONNX model and decoder with past ONNX model into one ONNX model with if logic.

    Args:
        decoder (`~onnx.ModelProto`): Decoder ONNX model.
        decoder_with_past (`~onnx.ModelProto`): Decoder with past ONNX model.
        graph_name (`~str`): Name of the parent graph(graph of the control flow node).
        producer_name (`~str`): Graph producer name.
        simplify_graph (`~bool`): Whether to use onnx-simplifier to simplify subgraphs.

    Returns:
        `~onnx.ModelProto`: The fused decoder ONNX model.
    """
    if simplify_graph:
        decoder = simplify_onnx_graph(decoder)
        decoder_with_past = simplify_onnx_graph(decoder_with_past)

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
                version=_get_onnx_opset(decoder),
            )
        ],
    )
    onnx.checker.check_model(merged_model)

    return merged_model
