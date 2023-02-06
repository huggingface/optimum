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

from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import numpy as np

import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper


def _find_duplicate_weights(model) -> DefaultDict[Tuple[int, bytes], Set[str]]:
    """
    TODO: short documentation.
    """
    return _find_duplicate_initializers(model.graph.initializer)


def _find_duplicate_initializers(initializers) -> DefaultDict[Tuple[int, bytes], Set[str]]:
    """
    TODO: short documentation.
    """
    duplicates = defaultdict(set)
    for initializer in initializers:
        tensor_dims = tuple(getattr(initializer, "dims"))
        for data_attr in ["raw_data", "int32_data", "int64_data", "uint64_data", "float_data", "double_data"]:
            tensor_data = getattr(initializer, data_attr)
            if tensor_data:
                tensor_data = tuple(tensor_data)
                break
        duplicates[(initializer.data_type, tensor_data, tensor_dims)].add(initializer.name)
    return duplicates


def _create_name_sharing_dict(
    duplicate_weights: DefaultDict[Tuple[int, bytes], Set[str]], suffix: str = None
) -> Dict[str, str]:
    """
    TODO: short documentation.
    """

    def _create_name_sharing_dict_for_duplicates(duplicates: Set[str]) -> Dict[str, str]:
        common_name = duplicates.pop()
        duplicates.add(common_name)
        if suffix:
            return {k: f"{common_name}_{suffix}" for k in duplicates}
        else:
            return {k: common_name for k in duplicates}

    name_sharing_dict = {}
    for duplicates in duplicate_weights.values():
        name_sharing_dict.update(_create_name_sharing_dict_for_duplicates(duplicates))
    return name_sharing_dict


def _replace_input_names(model: ModelProto, name_sharing_dict: Dict[str, str]):
    """
    TODO: short documentation.
    """
    for node in model.graph.node:
        for i in range(len(node.input)):
            node.input[i] = name_sharing_dict.get(node.input[i], node.input[i])


def _remove_redundant_initializers(model: ModelProto, name_sharing_dict: Dict[str, str]):
    """
    TODO: short documentation.
    """
    to_pop = []
    for idx, initializer in enumerate(model.graph.initializer):
        if initializer.name != name_sharing_dict[initializer.name]:
            to_pop.append(idx)

    for idx in sorted(to_pop, reverse=True):
        model.graph.initializer.pop(idx)


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
            raise ValueError(f"Cannot find `dim_param` nor `dim_value` in the output dimension info.")

    return output_shape


def _check_num_outputs(model1: ModelProto, model2: ModelProto):
    """
    Checks that `model1` and `model2` have the same number of outputs.
    """
    if not len(model1.graph.output) == len(model2.graph.output):
        raise ValueError(
            f"Two model protos need to have the same outputs. But one has {len(model1.graph.output)} "
            f"outputs while the other has {len(model2.graph.output)} outputs."
        )


def _unify_onnx_outputs(model1: ModelProto, model2: ModelProto):
    """
    Unifies the outputs of two ONNX model protos. The outputs of model1 will be replaced by outputs of model2.
    According to the rules of "If" op, two subgraphs must have the same number of outputs.
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
    all_initializers = []
    for model in models:
        all_initializers += list(model.graph.initializer)

    duplicates = _find_duplicate_initializers(all_initializers)
    name_sharing_dict = _create_name_sharing_dict(duplicates, suffix=suffix)
    for model in models:
        _replace_input_names(model, name_sharing_dict)

    deduplicated_initializers = []
    deduplicated_name = set()

    for initializer in all_initializers:
        if name_sharing_dict[initializer.name] not in deduplicated_name:
            deduplicated_name.add(name_sharing_dict[initializer.name])
            initializer.name = name_sharing_dict[initializer.name]
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
