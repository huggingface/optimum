#  Copyright 2022 The HuggingFace Team. All rights reserved.
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
import numpy as np
import torch

from onnxruntime.transformers.io_binding_helper import TypeHelper as ORTTypeHelper


# Adapted from https://github.com/microsoft/onnxruntime/blob/93e0a151177ad8222c2c95f814342bfa27f0a64d/onnxruntime/python/tools/transformers/io_binding_helper.py#L12
class TypeHelper(ORTTypeHelper):
    """
    Gets data type information of the ONNX Runtime inference session and provides the mapping from
    `OrtValue` data types to the data types of other frameworks (NumPy, PyTorch, etc).
    """

    # TODO: Current DLPack doesn't support boolean tensor, use uint8 as workaround, remove after it is supported.
    @staticmethod
    def ort_type_to_numpy_type(ort_type: str):
        ort_type_to_numpy_type_map = {
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bool)": np.uint8,
        }
        if ort_type in ort_type_to_numpy_type_map:
            return ort_type_to_numpy_type_map[ort_type]
        else:
            raise ValueError(
                f"{ort_type} is not supported. Here is a list of supported data type: {ort_type_to_numpy_type_map.keys()}"
            )

    @staticmethod
    def ort_type_to_torch_type(ort_type: str):
        ort_type_to_torch_type_map = {
            "tensor(int64)": torch.int64,
            "tensor(int32)": torch.int32,
            "tensor(int8)": torch.int8,
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
            "tensor(bool)": torch.bool,
        }
        if ort_type in ort_type_to_torch_type_map:
            return ort_type_to_torch_type_map[ort_type]
        else:
            raise ValueError(
                f"{ort_type} is not supported. Here is a list of supported data type: {ort_type_to_torch_type_map.keys()}"
            )
