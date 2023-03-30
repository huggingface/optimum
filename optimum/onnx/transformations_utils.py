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
from typing import DefaultDict, Dict, List, Set, Tuple

import numpy as np

import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper

from ..utils import logging


logger = logging.get_logger()
logger.setLevel(logging.INFO)


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
                " Constant outputs will be added to unify the two models outputs."
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
                    if not isinstance(dims_output_1[0], str):
                        raise ValueError(
                            f"Expected a dynamic shape for the axis zero of {model_output_1.name}, found a static shape: {dims_output_1[0]}"
                        )

                    # fill the constant shape with the original shape, except for the axis zero that is 0 for an empty constant,
                    # and the dynamic axis set to 1
                    dims_dummy_output = [0]
                    for dim in dims_output_1[1:]:
                        if isinstance(dim, str):
                            dims_dummy_output.append(1)
                        else:
                            dims_dummy_output.append(dim)

                    logger.info(
                        f"Addind a constant output for {model_output_1.name} of shape {dims_dummy_output} in model2."
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
