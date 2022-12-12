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
import logging
import traceback
from typing import TYPE_CHECKING

import numpy as np
import torch

import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.transformers.io_binding_helper import TypeHelper as ORTTypeHelper

from ..utils import is_cupy_available, is_onnxruntime_training_available


if TYPE_CHECKING:
    from ..modeling_ort import ORTModel

if is_cupy_available():
    import cupy as cp


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


# Adapted from https://github.com/microsoft/onnxruntime/blob/1ab11a111ce0717bfbfaca964d04a017cb9b1752/onnxruntime/python/tools/transformers/io_binding_helper.py#L97
class IOBindingHelper:
    """
    A helper class to enable `ORTModel` instances to prepare IO binding  with dynamic shaped outputs for an inference session and transfer
    tensors from ONNX Runtime to other frameworks on device. It helps reduce memory copy between the host and device.
    """

    def __init__(self, model: ort.InferenceSession, device, **kwargs):
        self.model = model
        self.device = device
        # Create {name:idx} dict for model inputs and outputs
        self.model_inputs = {output_key.name: idx for idx, output_key in enumerate(model.get_inputs())}
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(model.get_outputs())}
        self.model_input_names = list(self.model_inputs.keys())
        self.model_output_names = list(self.model_outputs.keys())

    @staticmethod
    def to_pytorch(ort_value: OrtValue) -> torch.Tensor:
        """
        Converts tensors held by OrtValues in ONNX runtime memory buffer to torch tensor.
        """

        if is_onnxruntime_training_available():
            return IOBindingHelper.to_pytorch_via_dlpack(ort_value)
        else:
            try:
                return IOBindingHelper.to_pytorch_via_cupy(ort_value)
            except Exception as e:
                logging.error(traceback.format_exc())
                logging.info("Unable to access output memory in CUDA, will offload to CPU")
                return IOBindingHelper.to_pytorch_via_numpy(ort_value)

    @staticmethod
    def to_pytorch_via_numpy(ort_value: OrtValue) -> torch.Tensor:
        ort_device = ort_value.device_name().lower()
        return torch.tensor(ort_value.numpy()).to(ort_device)

    @staticmethod
    def to_pytorch_via_cupy(ort_value: OrtValue) -> torch.Tensor:
        ort_device = ort_value.device_name().lower()
        if ort_device != "cuda":
            raise RuntimeError(f"Exchange tensors to PyTorch via CuPy only when device is CUDA, got: {ort_device}")

        ort_type = ort_value.data_type()
        numpy_type = TypeHelper.ort_type_to_numpy_type(ort_type)

        # Access CUDA memory via CuPy
        memory = cp.cuda.UnownedMemory(ort_value.data_ptr(), 0, None)
        memory_ptr = cp.cuda.MemoryPointer(memory, 0)
        cp_array = cp.ndarray(shape=ort_value.shape(), memptr=memory_ptr, dtype=numpy_type)
        torch_tensor = torch.from_dlpack(cp_array.toDlpack())

        # If is boolean, the dtype will be uint8 and need to be convert back to bool.
        if "bool" in ort_type:
            torch_tensor = torch_tensor.to(torch.bool)

        torch_tensor = torch_tensor.clone()

        return torch_tensor

    @staticmethod
    # dlpack support is available for OrtValue only when `onnxruntime-training` is installed
    def to_pytorch_via_dlpack(ort_value: OrtValue) -> torch.Tensor:
        from torch._C import _from_dlpack

        torch_tensor = ort_value.to_dlpacks(_from_dlpack)
        return torch_tensor

    @staticmethod
    def get_device_index(device):
        if isinstance(device, str):
            # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
            device = torch.device(device)
        elif isinstance(device, int):
            return device
        return 0 if device.index is None else device.index

    @staticmethod
    def prepare_io_binding(ort_model: "ORTModel", **inputs) -> ort.IOBinding:
        """
        Returns an IOBinding object for an inference session. This method is for general purpose, if the inputs and outputs
        are determined, you can prepare data buffers directly to avoid tensor transfers across frameworks.
        """
        if not all(input_name in inputs.keys() for input_name in ort_model.model_input_names):
            raise ValueError(
                f"The ONNX model takes {ort_model.model_input_names} as inputs, but only {inputs.keys()} are given."
            )

        name_to_np_type = TypeHelper.get_io_numpy_type_map(ort_model.model)

        # Bind inputs and outputs to onnxruntime session
        io_binding = ort_model.model.io_binding()

        # Bind inputs
        for input_name in ort_model.model_input_names:
            onnx_input = inputs.pop(input_name)
            onnx_input = onnx_input.contiguous()

            io_binding.bind_input(
                input_name,
                onnx_input.device.type,
                ort_model.device.index,
                name_to_np_type[input_name],
                list(onnx_input.size()),
                onnx_input.data_ptr(),
            )

        # Bind outputs
        for name in ort_model.model_output_names:
            io_binding.bind_output(name, ort_model.device.type, device_id=ort_model.device.index)

        return io_binding
