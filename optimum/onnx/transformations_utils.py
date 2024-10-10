#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple

import numpy as np

import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper


if TYPE_CHECKING:
    import torch.nn as nn

from ..utils import logging, recurse_getattr


logger = logging.get_logger()


def _find_duplicate_initializers(
    models: List[ModelProto],
) -> DefaultDict[Tuple[int, str, Tuple], Set[Tuple[str, int]]]:
    """
    Creates a map (unique data) --> set of (initializer name, model id)

    Initializers with a dimension 0, or dimension 1 with data type int32 or int64, are not included in the generated map.
    """
    duplicates = defaultdict(set)
    for i in range(len(models)):
        for initializer in models[i].graph.initializer:
            tensor_dims = tuple(getattr(initializer, "dims"))
            if len(tensor_dims) > 1 or (len(tensor_dims) == 1 and initializer.data_type not in [6, 7]):
                # Extract tensor data as numpy array
                tensor_data = numpy_helper.to_array(initializer)

                # Hash tensor data to avoid storing large amounts of data in memory
                hashed = hashlib.sha512()
                hashed.update(tensor_data)
                tensor_digest = hashed.hexdigest()

                duplicates[(initializer.data_type, tensor_digest, tensor_dims)].add((initializer.name, i))

    return duplicates


def _create_name_sharing_dict(
    duplicate_weights: DefaultDict[Tuple[int, str, Tuple], Set[Tuple[str, int]]], suffix: str = ""
) -> Dict[Tuple[str, int], str]:
    """
    Creates a map mapping old initializer names to new initializer names. As different ONNX models
    may use the same initializer name but need to be mapped to a different new name, the map is actually from
    (old name, model id) to new name.

    Example of initializers with the same name that will need to be mapped to a different one:
    Model 1 with:
    /transformer/Constant_8_output_0 of datatype 1

    Model 2 with:
    /transformer/Constant_8_output_0 of datatype 7

    Args:
        duplicate_weights (`DefaultDict[Tuple[int, bytes]`):

        suffix (`str`, defaults to `""`):
    """

    name_sharing_dict = {}
    used_common_names = {}
    for duplicates in duplicate_weights.values():
        common_name, model_id = duplicates.pop()

        # this is needed in case two different groups of shared initializers may share the same name, for example onnx::MatMul_2295 in the first
        # model, and onnx::MatMul_2295 in the second model, although point to different data
        if common_name in used_common_names:
            used_common_names[common_name] += 1
        else:
            used_common_names[common_name] = 0

        duplicates.add((common_name, model_id))
        for k in duplicates:
            assert k not in name_sharing_dict
            name_sharing_dict[k] = (
                f"{common_name}_{suffix}_{used_common_names[common_name]}" if suffix != "" else f"{common_name}"
            )

    return name_sharing_dict


def _replace_input_names(models: List[ModelProto], name_sharing_dict: Dict[Tuple[str, int], str]):
    """
    Replaces the names of node inputs from the models by the names in the name_sharing_dict.
    """
    for i in range(len(models)):
        for node in models[i].graph.node:
            for j in range(len(node.input)):
                if (node.input[j], i) in name_sharing_dict:
                    node.input[j] = name_sharing_dict[(node.input[j], i)]


def _remove_redundant_initializers(models: List[ModelProto], name_sharing_dict: Dict[Tuple[str, int], str]):
    """
    TODO: short documentation.
    """
    to_pop = []
    for i in range(len(models)):
        for idx, initializer in enumerate(models[i].graph.initializer):
            if initializer.name != name_sharing_dict[(initializer.name, i)]:
                to_pop.append(idx)

        for idx in sorted(to_pop, reverse=True):
            models[i].graph.initializer.pop(idx)


def _infer_output_shape(output: ValueInfoProto):
    """
    TODO: short documentation.
    """
    output_shape = []
    for dim in output.type.tensor_type.shape.dim:
        if getattr(dim, "dim_param"):
            output_shape.append(getattr(dim, "dim_param"))
        elif getattr(dim, "dim_value"):
            output_shape.append(getattr(dim, "dim_value"))
        else:
            raise ValueError("Cannot find `dim_param` nor `dim_value` in the output dimension info.")

    return output_shape


def _unify_onnx_outputs(model1: ModelProto, model2: ModelProto, strict: bool):
    """
    Unifies the outputs of two ONNX model protos. The outputs of model1 will be replaced by outputs of model2.
    According to the rules of "If" op, two subgraphs must have the same number of outputs.
    """

    model1_outputs = {output.name for output in model1.graph.output}
    model2_outputs = {output.name for output in model2.graph.output}

    if model1_outputs != model2_outputs:
        if strict is True:
            raise ValueError(
                f"The two model protos outputs are expected to have the same number of outputs and output names when strict=True. Found"
                f" the outputs {model1_outputs - model2_outputs} only in model1, and {model2_outputs - model1_outputs} only in model2."
            )
        else:
            logger.info(
                f"The two models proto have different outputs ({len(model1_outputs)} and {len(model2_outputs)} outputs)."
                " Constant outputs will be added to unify the two models outputs. This is expected for encoder-decoder models where cached cross-attention key/values are constant outputs, omitted in the model with KV cache."
            )

    if model2_outputs.issubset(model1_outputs) is False:
        raise ValueError("The second ModelProto should not have more outputs than the first.")

    for idx in range(len(model1.graph.output)):
        model_output_1 = model1.graph.output[idx]
        model_output_2 = model2.graph.output[idx] if idx < len(model2.graph.output) else None

        if model_output_2 is None or model_output_1 != model_output_2:
            if model_output_2 is None or not (
                model_output_1.name == model_output_2.name
                and model_output_1.type.tensor_type.elem_type == model_output_2.type.tensor_type.elem_type
            ):
                if strict is False and model_output_1.name not in model2_outputs:
                    data_type = model_output_1.type.tensor_type.elem_type
                    dims_output_1 = _infer_output_shape(model_output_1)
                    if not any(isinstance(dim_output, str) for dim_output in dims_output_1):
                        raise ValueError(
                            f"Expected at least one dynamic input shape for the output {model_output_1.name}, found a static shape: {dims_output_1}"
                        )

                    # fill the constant shape with the original shape, except for the first dynamic axis that is 0 for an empty constant,
                    # and the dynamic axis set to 1
                    dims_dummy_output = []
                    dummy_axis = None
                    for j, dim in enumerate(dims_output_1):
                        if isinstance(dim, str) and dummy_axis is None:
                            dims_dummy_output.append(0)
                            dummy_axis = j
                        elif isinstance(dim, str) and dummy_axis is not None:
                            dims_dummy_output.append(1)
                        else:
                            dims_dummy_output.append(dim)

                    logger.info(
                        f"Adding a constant output for {model_output_1.name} of shape {dims_dummy_output} in model2."
                    )
                    value = onnx.helper.make_tensor(
                        name="const_tensor", data_type=data_type, dims=dims_dummy_output, vals=[]
                    )
                    constant_node = onnx.helper.make_node(
                        "Constant",
                        name=f"Constant_{len(model2.graph.node) + 1}",
                        inputs=[],
                        outputs=[f"{model_output_1.name}"],
                        value=value,
                    )
                    model2.graph.node.append(constant_node)

                    constant_empty_output = onnx.helper.make_tensor_value_info(
                        model_output_1.name,
                        model_output_1.type.tensor_type.elem_type,
                        _infer_output_shape(model_output_1),
                    )
                    model2.graph.output.insert(idx, constant_empty_output)
                else:
                    if model_output_2 is not None:
                        raise ValueError(
                            f"Cannot match {model_output_1.name} with {model_output_2.name}. Make sure your"
                            f" model protos have same outputs, have same data types and are in the same order."
                        )
                    else:
                        raise ValueError(
                            f"Too few outputs of model2 were found to match with {model_output_1.name}."
                            f" Please try to pass strict=False, or fill a bug report at https://github.com/huggingface/optimum."
                        )
            else:
                model2.graph.output.remove(model_output_2)

                # We use model1 (normally the decoder) for the output shape
                # TODO: relax this, and keep the most permissive output shape between model1 and model2
                # while checking they are compatible
                new_output = onnx.helper.make_tensor_value_info(
                    model_output_1.name,
                    model_output_1.type.tensor_type.elem_type,
                    _infer_output_shape(model_output_1),
                )
                model2.graph.output.insert(idx, new_output)

    if not all(
        model_output_1 == model_output_2
        for model_output_1, model_output_2 in zip(model1.graph.output, model2.graph.output)
    ):
        raise RuntimeError("Failed to unify outputs of given ONNX model protos.")


def _get_all_inputs(model_list: List[ModelProto]) -> List[onnx.onnx_ml_pb2.ValueInfoProto]:
    """
    Returns all the inputs to all the models in `model_list`, in a single list.
    """
    inputs = []
    input_names = set()
    for model in model_list:
        for input in model.graph.input:
            if input.name not in input_names:
                input_names.add(input.name)
                inputs.append(input)
    return inputs


def _get_onnx_opset(model: ModelProto):
    """
    Returns the ONNX opset version used to generate `model`.
    """
    opset_import = model.opset_import[0]
    return getattr(opset_import, "version")


def _deduplicated_cross_model_initializers(models: List[ModelProto], suffix: str = None):
    """
    TODO: short documentation.
    """

    duplicates = _find_duplicate_initializers(models)
    name_sharing_dict = _create_name_sharing_dict(duplicates, suffix=suffix)

    _replace_input_names(models, name_sharing_dict)

    deduplicated_initializers = []
    deduplicated_name = set()

    for i in range(len(models)):
        for initializer in models[i].graph.initializer:
            name_id_pair = (initializer.name, i)
            if name_id_pair in name_sharing_dict and name_sharing_dict[name_id_pair] not in deduplicated_name:
                deduplicated_name.add(name_sharing_dict[name_id_pair])
                initializer.name = name_sharing_dict[name_id_pair]
                deduplicated_initializers.append(initializer)

    return deduplicated_initializers


def cast_int64_tensorproto_to_int32(initializer: onnx.TensorProto, cast: bool = False):
    """
    Casts in place the input TensorProto data to int32. Its data is assumed to be of type int64,
    and in case some values are out of range, they are cast to the min/max representable
    value in int32.
    """
    original_name = initializer.name
    array = np.copy(numpy_helper.to_array(initializer))

    if not array.dtype == np.int64:
        raise TypeError(
            "Expecting a `TensorProto` of type `int64` (represented as `7` in onnx.TensorProto) in the function int64_tensorproto_to_int32, but got {array.dtype}."
        )

    array[array > np.iinfo(np.int32).max] = np.iinfo(np.int32).max
    array[array < np.iinfo(np.int32).min] = np.iinfo(np.int32).min

    # the following line notably avoids the cast overhead in `convertOnnxWeights` in onnx-tensorrt
    if cast:
        array = array.astype(np.int32)
    array.setflags(write=0)

    tensor = numpy_helper.from_array(array)

    initializer.CopyFrom(tensor)
    initializer.name = original_name


def _get_weights_to_tie(tied_params: List[List[str]], torch_model: "nn.Module") -> Tuple[List[List[str]]]:
    """
    Separates tied weights from the torch_model in groups for which a tying implementation is (and is not) available.

    Currently, only Embedding and Linear weight sharing the same data can be tied.
    """
    SUPPORTED_DEDUPLICATION_OPS = ("Embedding", "Linear")
    tied_params_with_op = []
    tied_groups_to_tie = []
    tied_groups_ignored = []
    for params in tied_params:
        tied_params_with_op.append({})
        skip_group = False
        for param_name in params:
            module_name = ".".join(param_name.split(".")[:-1])

            module = recurse_getattr(torch_model, module_name)
            if module.__class__.__name__ not in SUPPORTED_DEDUPLICATION_OPS:
                skip_group = True

            tied_params_with_op[-1][param_name] = module.__class__.__name__

        if skip_group:
            tied_groups_ignored.append(params)
        else:
            tied_groups_to_tie.append(params)

    return tied_params_with_op, tied_groups_to_tie, tied_groups_ignored


def _find_matching_initializers(
    tied_params_with_op: List[Dict[str, str]], model: ModelProto, initializer_name_to_idx: Dict[str, int]
):
    """
    From the torch parameter names in `tied_params`, find the matching initializers
    in the ONNX model.

    Args:
        tied_params_with_op (`List[Dict[str, str]]`):
            A list of groups of parameters that are tied, i.e. shared. For them,
            the torch module share the same pointer. The dictionary points to what type of nn.Module the parameter belongs to (e.g. `Linear`).
        model (`ModelProto`):
            The model in which the initializers should be looked for.
        initializer_name_to_idx (`Dict[str, int]`):
            A mapping from the model initializer name to their indices in model.graph.initializer, to ease the search.

    Returns:
        tied_groups_map (`Dict[Tuple[str], List[Dict[str, Any]]]`):
            A mapping from a tied weight group to the list of tied parameters torch name and potentially matching initializers (several in case it could not be exactly found).
    """
    tied_groups_map = {}
    for params in tied_params_with_op:
        torch_to_initializer = []
        for param_name, torch_op_name in params.items():
            # To find which initializer correspond to a torch parameter, we first look for
            # exactly matching initializer name.
            identical_initializer = False
            if param_name in initializer_name_to_idx.keys():
                nodes_containing_initializer = set()
                for node in model.graph.node:
                    if param_name in node.input:
                        nodes_containing_initializer.add(node.name)

                torch_to_initializer.append(
                    {
                        "param_name": param_name,
                        "initializer_name": {param_name},
                        "nodes_containing_initializer": nodes_containing_initializer,
                    }
                )
                identical_initializer = True

            # If not found (e.g. "lm_head.weight"), we greedily search for all initializers from potentially matching node names (e.g. "lm_head"),
            # or e.g. for predictions.decoder.weight search any of *predictions/decoder*
            # This greedy approach may found more initializers than wanted.
            if not identical_initializer:
                module_name = "/".join(param_name.split(".")[:-1])

                if param_name.endswith("weight") and torch_op_name == "Linear":
                    module_name += "/MatMul"
                elif param_name.endswith("bias") and torch_op_name == "Linear":
                    module_name += "/Add"

                candidate_inputs = {}
                candidate_node_idxs = []
                for i, node in enumerate(model.graph.node):
                    if module_name in node.name:
                        candidate_node_idxs.append(i)

                for node_idx in candidate_node_idxs:
                    node_name = model.graph.node[node_idx].name
                    candidate_inputs[node_name] = list(model.graph.node[node_idx].input)
                torch_to_initializer_param = set()
                nodes_containing_initializer = set()
                for node_name, input_names in candidate_inputs.items():
                    for input_name in input_names:
                        if input_name in initializer_name_to_idx.keys():
                            torch_to_initializer_param.add(input_name)
                            nodes_containing_initializer.add(node_name)

                if len(torch_to_initializer_param) == 0:
                    logger.warning(
                        f"Could not find ONNX initializer for torch parameter {param_name}. {param_name} will not be checked for deduplication."
                    )

                torch_to_initializer.append(
                    {
                        "param_name": param_name,
                        "initializer_name": torch_to_initializer_param,
                        "nodes_containing_initializer": nodes_containing_initializer,
                    }
                )

        intersect = torch_to_initializer[0]["initializer_name"]
        for i in range(1, len(params)):
            intersect = intersect.intersection(torch_to_initializer[i]["initializer_name"])

        if len(intersect) == 0:
            logger.warning("Found different candidate ONNX initializers (likely duplicate) for the tied weights:")
            not_found = []
            for i, torch_to_onnx_map in enumerate(torch_to_initializer):
                warn_string = f"\t{torch_to_onnx_map['param_name']}: {torch_to_onnx_map['initializer_name']}"
                if len(torch_to_onnx_map["initializer_name"]) == 0:
                    not_found.append(i)
                    warn_string += " --> ignored (may be a parameter from a part of the model not exported)"
                logger.warning(warn_string)

            # There may be some parameters in a tied group that are not present in the ONNX. That is for example the case in encoder-decoder
            # models where a tied parameter as model.encoder.embed_tokens.weight is detected even for the decoder model.
            for index in not_found[::-1]:
                del torch_to_initializer[index]

            if any(len(torch_to_onnx_map["initializer_name"]) > 1 for torch_to_onnx_map in torch_to_initializer):
                logger.warning(
                    f"Could not find unique initializers corresponding to the torch tied parameters {params}. Deduplication will be skipped for this group of weights although it should be done. Please open an issue in Optimum repository."
                )
                continue

        tied_groups_map[tuple(params)] = torch_to_initializer
    return tied_groups_map


def _deduplicate_gather_matmul(
    model: ModelProto,
    tied_groups_to_tie: List[List[str]],
    tied_groups_map: Dict[Tuple[str], List[Dict[str, Any]]],
    initializer_name_to_idx: Dict[str, int],
):
    """
    Removes the duplicate initializers for Gather and MatMul from the ONNX model based on the information in tied_groups_map i.e. of which ONNX initializers correspond to a single torch parameter.
    """
    node_name_to_idx = {}
    for idx, node in enumerate(model.graph.node):
        node_name_to_idx[node.name] = idx

    for params in tied_groups_to_tie:
        torch_to_initializer = tied_groups_map[tuple(params)]

        # ONNX Runtime quantization behaves bad with Transpose -> Gather. Thus, we take as reference the Gather node, and rather edit MatMul nodes.
        ref_idx = None
        for i in range(len(torch_to_initializer)):
            ops_using_initializer = set()
            for node_name in torch_to_initializer[i]["nodes_containing_initializer"]:
                ops_using_initializer.add(model.graph.node[node_name_to_idx[node_name]].op_type)

            if ops_using_initializer == {"Gather"}:
                ref_idx = i
                break

        if ref_idx is None:
            logger.warning(
                f"Could not deduplicate initializers corresponding to the torch tied parameters {params} as an initializer used only by Gather nodes could not be found. Skipping deduplication."
            )
            continue

        ref_initializer_name = next(iter(torch_to_initializer[ref_idx]["initializer_name"]))
        ref_initializer_idx = initializer_name_to_idx[ref_initializer_name]
        ref_initializer = model.graph.initializer[ref_initializer_idx]
        ref_type = ref_initializer.data_type
        ref_data = numpy_helper.to_array(ref_initializer)

        for i in range(len(torch_to_initializer)):
            if i == ref_idx:
                continue

            initializer_name = next(iter(torch_to_initializer[i]["initializer_name"]))
            initializer_idx = initializer_name_to_idx[initializer_name]
            initializer = model.graph.initializer[initializer_idx]
            initializer_type = initializer.data_type
            initializer_data = numpy_helper.to_array(initializer)

            # Several torch parameters may correspond to the same initializer.
            if initializer_name == ref_initializer_name:
                continue

            if ref_type == initializer_type and np.array_equal(ref_data, initializer_data):
                # The duplicate initializer are exactly identical
                logger.info(f"Removing duplicate initializer {initializer_name}...")

                # Change initializer to the reference initializer
                for node in model.graph.node:
                    if initializer_name in node.input:
                        input_idx = list(node.input).index(initializer_name)
                        node.input[input_idx] = ref_initializer_name

                # Remove old initializer
                model.graph.initializer.pop(initializer_idx)
            elif ref_type == initializer_type and np.array_equal(ref_data.T, initializer_data):
                # The duplicate initializer is the ref transposed
                logger.info(f"Removing duplicate initializer {initializer_name}...")

                # Add transpose node at the correct position to keep the topological order
                transpose_output_name = f"{ref_initializer_name}_transposed"
                transpose_node_name = f"Transpose_{len(model.graph.node) + 1}"

                minimum_node_idx = len(model.graph.node)
                for node_idx, node in enumerate(model.graph.node):
                    if initializer_name in node.input:
                        minimum_node_idx = node_idx
                        break

                transpose_node = onnx.helper.make_node(
                    "Transpose",
                    name=transpose_node_name,
                    inputs=[ref_initializer_name],
                    outputs=[transpose_output_name],
                )
                model.graph.node.insert(minimum_node_idx, transpose_node)

                # Change initializer to transpose output
                for node in model.graph.node:
                    if initializer_name in node.input:
                        input_idx = list(node.input).index(initializer_name)
                        node.input[input_idx] = transpose_output_name

                # Remove old initializer
                model.graph.initializer.pop(initializer_idx)
            else:
                logger.warning(
                    f"No deduplication implementation for {initializer_name} although it should be deduplicated. Please open an issue in Optimum repository."
                )
    return model
