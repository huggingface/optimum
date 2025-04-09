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
"""Defines the base classes that are used to perform inference with ONNX Runtime of Transformers models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from onnxruntime import InferenceSession, IOBinding, SessionOptions
from onnxruntime.transformers.io_binding_helper import TypeHelper

from ..utils.logging import get_logger
from .utils import (
    get_device_for_provider,
    get_dtype_from_session,
    get_provider_for_device,
    parse_device,
    validate_provider_availability,
)


logger = get_logger(__name__)

NON_EMPTY_TENSOR = torch.tensor(0)

ORTMIXIN_END_DOCSTRING = r"""
    This model inherits from [`~onnxruntime.base.ORTMixin`], check its documentation for the generic methods the
    library implements for all its model (such as onnx input/output preparation, device management, i/o binding, etc.).
"""


class ORTMixin:
    def init_ort_attributes(self, session: InferenceSession, use_io_binding: Optional[bool] = None):
        """
        Initializes the ONNX Runtime attributes of the loaded session.
        """

        self.session = session

        if use_io_binding is None:
            if self.provider == "CUDAExecutionProvider":
                logger.info(
                    "`use_io_binding` was not set, but CUDAExecutionProvider supports IO Binding. "
                    "Setting `use_io_binding=True` to leverage IO Binding and improve performance. "
                    "You can disable it by setting `model.use_io_binding=False`."
                )
                use_io_binding = True
            else:
                use_io_binding = False

        self._io_binding = IOBinding(session)
        self._use_io_binding = use_io_binding
        self._device = get_device_for_provider(self.provider, self.provider_option)
        self._dtype = get_dtype_from_session(session)

        # TODO: we can get rid of these as well (accessible from session)
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(session.get_inputs())}
        self.input_shapes = {input_key.name: input_key.shape for input_key in session.get_inputs()}
        self.input_dtypes = {input_key.name: input_key.type for input_key in session.get_inputs()}

        self.output_names = {output_key.name: idx for idx, output_key in enumerate(session.get_outputs())}
        self.output_shapes = {output_key.name: output_key.shape for output_key in session.get_outputs()}
        self.output_dtypes = {output_key.name: output_key.type for output_key in session.get_outputs()}

    @property
    def providers(self) -> List[str]:
        """
        Returns a list of Execution Providers registered with the session.
        """
        return self.session.get_providers()

    @property
    def provider(self) -> str:
        """
        Returns the main Execution Provider registered with the session.
        """
        return self.providers[0]

    @property
    def provider_options(self) -> Dict[str, Any]:
        """
        Returns a dictionary of Execution Providers configurations/options.
        """
        return self.session.get_provider_options()

    @property
    def provider_option(self) -> Dict[str, Any]:
        """
        Returns the configuration/options of the main Execution Provider.
        """
        return self.provider_options[self.provider]

    @property
    def device(self) -> torch.device:
        """
        Returns the `torch.device` associated with the ONNX Runtime session.
        This device is inferred from the provider and provider options.
        """
        return self._device

    @device.setter
    def device(self, *args, **kwargs):
        raise AttributeError(
            "The device attribute is read-only, please use the `.to(device)` "
            "method to change both the device and the execution provider accordingly."
        )

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the `torch.dtype` associated with the ONNX Runtime session.
        This dtype is inferred from the input/output dtypes of the session.
        If no floating point type is found, it defaults to `torch.float32`.
        """
        return self._dtype

    @property
    def use_io_binding(self) -> Optional[bool]:
        """
        Returns whether IO Binding is used or not.
        """
        return self._use_io_binding

    @use_io_binding.setter
    def use_io_binding(self, value: bool):
        """
        Sets the IO Binding usage.
        """
        if not isinstance(value, bool):
            raise ValueError("`use_io_binding` should be a boolean value.")
        self._use_io_binding = value

    def to(self, *args, **kwargs):
        # extract device and dtype from args and kwargs
        dtype = None
        device = None

        for arg in args:
            if isinstance(arg, str):
                device = arg
            elif isinstance(arg, int):
                device = torch.device(arg)
            elif isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        for key, value in kwargs.items():
            if key == "device":
                device = value
            elif key == "dtype":
                dtype = value

        if dtype is not None:
            # we don't support changing the dtype of the model
            return self

        if device is None:
            # we don't support changing the device of the model
            return self

        device, provider_option = parse_device(device)
        provider = get_provider_for_device(device)
        validate_provider_availability(provider)

        if device == self.device:
            logger.info(f"Model is already on {device}. No action taken.")
            return self

        self.session.set_providers([provider], provider_options=[provider_option])

        if self.use_io_binding is None:
            if self.provider == "CUDAExecutionProvider":
                logger.info(
                    "`use_io_binding` was set to `None` before the provider was changed to CUDAExecutionProvider. "
                    "Setting `use_io_binding=True` to leverage IO Binding and improve performance. "
                    "You can disable it by setting `model.use_io_binding=False`."
                )
                self.use_io_binding = True

        self._device = device

        return self

    @staticmethod
    def load_model(
        path: Union[str, Path],
        provider: str = "CPUExecutionProvider",
        session_options: Optional[SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> InferenceSession:
        """
        Loads an ONNX Inference session with a given provider. Default provider is `CPUExecutionProvider` to match the
        default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            path (`Union[str, Path]`):
                Path of the ONNX model.
            provider (`str`, defaults to `"CPUExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[onnxruntime.SessionOptions]`, defaults to `None`):
                ONNX Runtime session options to use for loading the model.
            provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html .
        """
        validate_provider_availability(provider)  # raise error if the provider is not available

        providers = [provider]

        if provider_options is not None:
            provider_options = [provider_options]

        return InferenceSession(
            path, providers=providers, sess_options=session_options, provider_options=provider_options
        )

    def raise_on_numpy_input_io_binding(self, use_torch: bool):
        """
        Raises an error if IO Binding is requested although the tensor used are numpy arrays.

        Args:
            use_torch (`bool`):
                Whether the tensor used during inference are of type torch.Tensor or not.
        """
        if use_torch is False and self.use_io_binding is True:
            raise ValueError(
                "IO Binding can not be used when passing numpy inputs. Please disable IO Binding"
                " with `model.use_io_binding=False`, or pass `torch.Tensor` inputs instead."
            )

    def _prepare_onnx_inputs(
        self, use_torch: bool, model_inputs: Dict[str, Union[torch.Tensor, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Prepares the inputs for ONNX Runtime by converting them to numpy arrays with the expected dtype.

        Args:
            use_torch (`bool`):
                Whether the inputs are torch.Tensor or not.
            inputs (`Dict[str, Union[torch.Tensor, np.ndarray]]`):
                The inputs to prepare for ONNX Runtime.

        Returns:
            `Dict[str, np.ndarray]`: The inputs prepared for ONNX Runtime.
        """

        onnx_inputs = {}

        for input_name in self.input_names.keys():
            if model_inputs.get(input_name, None) is None:
                raise ValueError(f"Input {input_name} is required by model but not provided.")

            if use_torch:
                onnx_inputs[input_name] = model_inputs[input_name].numpy(force=True)
            else:
                onnx_inputs[input_name] = model_inputs[input_name]

            expected_dtype = TypeHelper.ort_type_to_numpy_type(self.input_dtypes[input_name])

            if onnx_inputs[input_name].dtype != expected_dtype:
                onnx_inputs[input_name] = onnx_inputs[input_name].astype(expected_dtype)

        return onnx_inputs

    def _prepare_onnx_outputs(
        self, use_torch: bool, onnx_outputs: List[np.ndarray]
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Prepares the outputs from ONNX Runtime by converting them to torch.Tensor if requested.

        Args:
            use_torch (`bool`):
                Whether the outputs should be torch.Tensor or not.
            onnx_outputs (`List[np.ndarray]`):
                The outputs from ONNX Runtime.

        Returns:
            `Dict[str, Union[torch.Tensor, np.ndarray]]`: The outputs prepared for the user.
        """

        model_outputs = {}

        for output_name, idx in self.output_names.items():
            model_outputs[output_name] = onnx_outputs[idx]

            if use_torch:
                model_outputs[output_name] = torch.from_numpy(model_outputs[output_name]).to(self.device)

        return model_outputs

    def _prepare_output_buffer(self, output_shape: Tuple[int], output_dtype: torch.dtype) -> torch.Tensor:
        """
        Prepares an output buffer for ONNX Runtime IO Binding.

        Args:
            output_shape (`Tuple[int]`):
                The shape of the output buffer.
            output_dtype (`torch.dtype`):
                The dtype of the output buffer.

        Returns:
            `torch.Tensor`: The output buffer.

        """
        assert len(output_shape) > 0, "`output_shape` should not be empty."
        assert all(
            isinstance(dim, int) for dim in output_shape
        ), f"`output_shape` should only contain integers but got {output_shape}."
        assert all(
            dim > 0 for dim in output_shape
        ), f"`output_shape` should only contain positive integers but got {output_shape}."

        if len(output_shape) > 0:
            output_buffer = torch.empty(np.prod(output_shape), dtype=output_dtype, device=self.device)
        else:
            output_buffer = torch.tensor(0, dtype=output_dtype, device=self.device)

        return output_buffer

    def _output_shape_inference(self, output_name: str, known_axes_values: Dict[str, int]) -> List[int]:
        """
        Infers the shape of a given output by using the `known_axes_values` mapping.

        Args:
            output_name (`str`):
                The name of the output for which to infer the shape.
            known_axes_values (`Dict[str, int]`):
                A mapping of the axis names to their values.

        Returns:
            `List[int]`: The inferred shape of the output.
        """

        output_shape = list(self.output_shapes[output_name])

        for idx, axis_name in enumerate(output_shape):
            if isinstance(axis_name, str):
                output_shape[idx] = self._dynamic_axis_inference(axis_name, known_axes_values)

        return output_shape

    def _dynamic_axis_inference(self, axis_name: Union[str], known_axes_values: Dict[str, int]) -> int:
        """
        Infers the value of a given dynamic axis by using the `known_axes_values` mapping.

        For instance, for the following inputs:
            axis_name = "sequence_length + past_sequence_length"
            known_axes_values = {"batch_size": 2, "sequence_length": 3, "past_sequence_length": 7}

        The inferred value will be:
            3 + 7 = 10
        """

        if axis_name in known_axes_values:
            # simple case, the axis value is known
            return known_axes_values[axis_name]

        tokens = axis_name.split(" ")
        for idx, token in enumerate(tokens):
            if token in known_axes_values:
                tokens[idx] = str(known_axes_values[token])

        return int(eval(" ".join(tokens)))

    def _prepare_io_binding(
        self,
        model_inputs: Dict[str, torch.Tensor],
        outputs_to_not_bind: Optional[Set[str]] = None,
        known_output_shapes: Optional[Dict[str, Tuple[int]]] = None,
    ) -> Tuple[Dict[str, Tuple[int]], Dict[str, torch.Tensor]]:
        """
        Prepares IO binding for ONNX Runtime.

        Args:
            model_inputs (`Dict[str, torch.Tensor]`):
                The inputs to bind to the model.
            outputs_to_not_bind (`Optional[Union[Set[str], str]]`, defaults to `None`):
                The names of the outputs that should not be bound.
            known_output_shapes (`Optional[Dict[str, Tuple[int]]]`, defaults to `None`):
                It can be hard to infer all the output shapes from the inputs only. For instance for the past key /
                values. It is possible to explicitely pass the shape via this argument.

        Returns:
            `TupleDict[str, Tuple[int]], Dict[str, torch.Tensor]`: A dictionary of the output shapes and a dictionary of
            the output buffers.
        """

        known_axes_values = {}

        for input_name in self.input_names.keys():
            input_shape = model_inputs[input_name].shape

            if not model_inputs[input_name].is_contiguous():
                model_inputs[input_name] = model_inputs[input_name].contiguous()

            tensor_dtype = model_inputs[input_name].dtype
            expected_dtype = TypeHelper.ort_type_to_torch_type(self.input_dtypes[input_name])
            if tensor_dtype != expected_dtype:
                model_inputs[input_name] = model_inputs[input_name].to(expected_dtype)

            data_ptr = model_inputs[input_name].data_ptr()
            if data_ptr == 0:
                # During first generation, sequence_length can be 0 when use_cache=True, which results in data_ptr to also be 0.
                # To keep compatibility with IO binding, we pass the data pointer of a non-empty tensor.
                # No impact because past_key_values will not be used during the first generation.
                data_ptr = NON_EMPTY_TENSOR.data_ptr()

            self._io_binding.bind_input(
                input_name,
                self.device.type,
                self.device.index or 0,
                TypeHelper.ort_type_to_numpy_type(self.input_dtypes[input_name]),
                input_shape,
                data_ptr,
            )

            for idx, axis_name in enumerate(self.input_shapes[input_name]):
                if isinstance(axis_name, str):
                    known_axes_values[axis_name] = input_shape[idx]

        output_shapes = {}
        output_buffers = {}

        for output_name in self.output_names.keys():
            if outputs_to_not_bind is not None and output_name in outputs_to_not_bind:
                continue

            if known_output_shapes is not None and output_name in known_output_shapes:
                output_shape = known_output_shapes[output_name]
            else:
                output_shape = self._output_shape_inference(output_name, known_axes_values)

            output_dtype = TypeHelper.ort_type_to_torch_type(self.output_dtypes[output_name])
            output_buffer = self._prepare_output_buffer(output_shape, output_dtype)
            data_ptr = output_buffer.data_ptr()

            self._io_binding.bind_output(
                output_name,
                self.device.type,
                self.device.index or 0,
                TypeHelper.ort_type_to_numpy_type(self.output_dtypes[output_name]),
                output_shape,
                data_ptr,
            )

            output_buffers[output_name] = output_buffer
            output_shapes[output_name] = output_shape

        return output_shapes, output_buffers

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The `forward` method should be implemented in the derived class. "
            "Please refer to the documentation for more details."
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTMultiPartWrapper:
    """
    Wrapper class for multiple ORTMixin models. This class allows to combine multiple models into a single
    inference pipeline. It is useful for models that require multiple parts to work together, such as
    Diffusion pipelines or encoder-decoder models, as it provides a unified interface for inference.
    """

    def init_ort_wrapper_attributes(self, parts: List[ORTMixin]):
        self.parts = parts

        if len(self.parts) < 1:
            raise ValueError("ORTMultiPartWrapper should be initialized with at least one ORTMixin.")

        if any(not isinstance(model, ORTMixin) for model in self.parts):
            raise ValueError("All parts in ORTMultiPartWrapper should be of type ORTMixin.")

    @property
    def use_io_binding(self):
        """
        Returns whether IO Binding is used or not.
        """
        for model in self.parts:
            return model.use_io_binding

    @use_io_binding.setter
    def use_io_binding(self, value: bool):
        """
        Setter for the use_io_binding property.
        """
        for model in self.parts:
            model.use_io_binding = value

    @property
    def providers(self):
        """
        Returns a list of Execution Providers registered with the session.
        """
        for model in self.parts:
            return model.providers

    @property
    def provider(self):
        """
        Returns the main Execution Provider registered with the session.
        """
        for model in self.parts:
            return model.provider

    @property
    def provider_options(self):
        """
        Returns a dictionary of Execution Providers configurations/options.
        """
        for model in self.parts:
            return model.provider_options

    @property
    def provider_option(self):
        """
        Returns the configuration/options of the main Execution Provider.
        """
        for model in self.parts:
            return model.provider_option

    @property
    def device(self):
        """
        Returns the `torch.device` associated with the ONNX Runtime session.
        This device is inferred from the provider and provider options.
        """
        for model in self.parts:
            return model.device

    @property
    def dtype(self):
        """
        Returns the `torch.dtype` associated with the ONNX Runtime session.
        This dtype is inferred from the input/output dtypes of the session.
        If no floating point type is found, it defaults to `torch.float32`.
        """
        for model in self.parts:
            return model.dtype

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device and/or dtype.
        """

        for model in self.parts:
            model.to(*args, **kwargs)

        return self
