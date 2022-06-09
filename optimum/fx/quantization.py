# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.fx.node import Argument, Node, Target
from torch.nn.intrinsic import _FusedModule
from torch.quantization.fx.graph_module import GraphModule, ObservedGraphModule
from torch.quantization.quantize_fx import Scope, ScopeContextManager
from torch.quantization.quantize_fx import fuse_fx as orig_fuse_fx
from torch.quantization.quantize_fx import prepare_fx as orig_prepare_fx
from torch.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx
from transformers.utils.fx import HFTracer, check_if_model_is_supported, get_concrete_args, symbolic_trace

from .utils import check_if_available


if TYPE_CHECKING:
    from torch.fx import Graph
    from transformers import PreTrainedModel


class QuantizationTracer(HFTracer):
    """
    Transformers compatible version of torch.quantization.quantize_fx.QuantizationTracer.
    This tracer is used internally to prepare the model for quantization.
    """
    specialized_concrete_args: Optional[Dict[str, Any]] = None

    def __init__(self, skipped_module_names: List[str], skipped_module_classes: List[Callable]):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type of top level
        # module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}

    def is_leaf_module(self, module: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            (module.__module__.startswith("torch.nn") and not isinstance(module, torch.nn.Sequential))
            or module_qualified_name in self.skipped_module_names
            or type(module) in self.skipped_module_classes
            or isinstance(module, _FusedModule)
            or super().is_leaf_module(module, module_qualified_name)
        )

    def call_module(
        self, module: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        module_qualified_name = self.path_of_module(module)
        # Creating scope with information of current module
        # scope will be restored automatically upon exit
        with ScopeContextManager(self.scope, module, module_qualified_name):
            return super().call_module(module, forward, args, kwargs)

    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (self.scope.module_path, self.scope.module_type)
        return node

    def trace(self, root: "PreTrainedModel", concrete_args: Optional[Dict[str, Any]] = None) -> "Graph":
        if concrete_args is None and self.specialized_concrete_args is not None:
            concrete_args = self.specialized_concrete_args
        return super().trace(root, concrete_args=concrete_args)


def specialized_quantization_tracer_creator(concrete_args: Dict[str, Any]) -> Type:
    """Creates a QuantizationTracer-like class specifying concrete_args as a class attribute."""
    return type("QuantizationTracer", (QuantizationTracer,), {"specialized_concrete_args": concrete_args})


@check_if_available
def fuse_fx(
    model: Union["PreTrainedModel", GraphModule],
    fuse_custom_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    check: bool = True,
) -> GraphModule:
    """
    Transformers models compatible version of torch.quantization.quantize_fx.fuse_fx, refer to PyTorch documentation
    for more details: https://pytorch.org/docs/stable/generated/torch.quantization.quantize_fx.fuse_fx.html.

    Args:
        model (`PreTrainedModel` or `torch.fx.GraphModule`):
            The model to fuse.
        fuse_custom_config_dict (`Dict[str, Any]`, *optional*):
            Dictionary for custom configurations for fuse_fx, e.g.:

                ```python
                >>> fuse_custom_config_dict = {
                >>>   "additional_fuser_method_mapping": {
                >>>     (Module1, Module2): fuse_module1_module2
                >>>   }

                >>>   # Attributes that are not used in forward function will
                >>>   # be removed when constructing GraphModule, this is a list of attributes
                >>>   # to preserve as an attribute of the GraphModule even when they are
                >>>   # not used in the code, these attributes will also persist through deepcopy
                >>>   "preserved_attributes": ["preserved_attr"],
                >>> }
                ```

        input_names (`List[str]`, *optional*):
            The input names of the model, only used to trace if model is a PreTrainedModel. This is not needed if model
            is already a GraphModule.
        check (`bool`, *optional*, defaults to `True`):
            If True, a check is done to verify that the model can be traced.

    Returns:
        `torch.fx.GraphModule`: A GraphModule with the fused modules.

    Example:

        ```python
        >>> from torch.ao.quantization import fuse_fx
        >>> m = Model().eval()
        >>> m = fuse_fx(m)
        ```
    """
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        model = symbolic_trace(model, input_names, disable_check=not check)
    orig_symbolic_trace = torch.fx.symbolic_trace
    torch.fx.symbolic_trace = lambda x: x
    graph_module = orig_fuse_fx(model, fuse_custom_config_dict=fuse_custom_config_dict)
    torch.fx.symbolic_trace = orig_symbolic_trace
    return graph_module


@check_if_available
def prepare_fx(
    model: Union["PreTrainedModel", GraphModule],
    qconfig_dict: Any,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
    equalization_qconfig_dict: Optional[Dict[str, Any]] = None,
    backend_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    check: bool = True,
) -> ObservedGraphModule:
    """
    Transformers models compatible version of torch.quantization.quantize_fx.prepare_fx, refer to PyTorch documentation
    for more details: https://pytorch.org/docs/stable/generated/torch.quantization.quantize_fx.prepare_fx.html#torch.quantization.quantize_fx.prepare_fx.

    Args:
        model (`PreTrainedModel` or `torch.fx.GraphModule`):
            The model to prepare, must be in eval mode.
        qconfig_dict (`Any`):
             qconfig_dict is a dictionary with the following configurations:

                ```python
                >>> qconfig_dict = {
                >>>   # optional, global config
                >>>   "": qconfig?,

                >>>   # optional, used for module and function types
                >>>   # could also be split into module_types and function_types if we prefer
                >>>   "object_type": [
                >>>     (torch.nn.Conv2d, qconfig?),
                >>>     (torch.nn.functional.add, qconfig?),
                >>>     ...,
                >>>    ],

                >>>   # optional, used for module names
                >>>   "module_name": [
                >>>     ("foo.bar", qconfig?)
                >>>     ...,
                >>>   ],

                >>>   # optional, matched in order, first match takes precedence
                >>>   "module_name_regex": [
                >>>     ("foo.*bar.*conv[0-9]+", qconfig?)
                >>>     ...,
                >>>   ],

                >>>   # optional, used for matching object type invocations in a submodule by
                >>>   # order
                >>>   # TODO(future PR): potentially support multiple indices ('0,1') and/or
                >>>   #   ranges ('0:3').
                >>>   "module_name_object_type_order": [
                >>>     # fully_qualified_name, object_type, index, qconfig
                >>>     ("foo.bar", torch.nn.functional.linear, 0, qconfig?),
                >>>   ],

                >>>   # priority (in increasing order):
                >>>   #   global, object_type, module_name_regex, module_name,
                >>>   #   module_name_object_type_order
                >>>   # qconfig == None means fusion and quantization should be skipped for anything
                >>>   # matching the rule
                >>> }
                ```

        prepare_custom_config_dict (`Dict[str, Any]`, *optional*):
            Customization configuration dictionary for quantization tool:

            ```python
            >>> prepare_custom_config_dict = {
            >>>   # optional: specify the path for standalone modules
            >>>   # These modules are symbolically traced and quantized as one unit
            >>>   "standalone_module_name": [
            >>>      # module_name, qconfig_dict, prepare_custom_config_dict
            >>>      ("submodule.standalone",
            >>>       None,  # qconfig_dict for the prepare function called in the submodule,
            >>>              # None means use qconfig from parent qconfig_dict
            >>>       {"input_quantized_idxs": [], "output_quantized_idxs": []}),  # prepare_custom_config_dict
            >>>       {}  # backend_config_dict, TODO: point to README doc when it's ready
            >>>   ],

            >>>   "standalone_module_class": [
            >>>       # module_class, qconfig_dict, prepare_custom_config_dict
            >>>       (StandaloneModule,
            >>>        None,  # qconfig_dict for the prepare function called in the submodule,
            >>>               # None means use qconfig from parent qconfig_dict
            >>>       {"input_quantized_idxs": [0], "output_quantized_idxs": [0]},  # prepare_custom_config_dict
            >>>       {})  # backend_config_dict, TODO: point to README doc when it's ready
            >>>   ],

            >>>   # user will manually define the corresponding observed
            >>>   # module class which has a from_float class method that converts
            >>>   # float custom module to observed custom module
            >>>   # (only needed for static quantization)
            >>>   "float_to_observed_custom_module_class": {
            >>>      "static": {
            >>>          CustomModule: ObservedCustomModule
            >>>      }
            >>>   },

            >>>   # the qualified names for the submodule that are not symbolically traceable
            >>>   "non_traceable_module_name": [
            >>>      "non_traceable_module"
            >>>   ],

            >>>   # the module classes that are not symbolically traceable
            >>>   # we'll also put dynamic/weight_only custom module here
            >>>   "non_traceable_module_class": [
            >>>      NonTraceableModule
            >>>   ],

            >>>   # Additional fuser_method mapping
            >>>   "additional_fuser_method_mapping": {
            >>>      (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
            >>>   },

            >>>   # Additioanl module mapping for qat
            >>>   "additional_qat_module_mapping": {
            >>>      torch.nn.intrinsic.ConvBn2d: torch.nn.qat.ConvBn2d
            >>>   },

            >>>   # Additional fusion patterns
            >>>   "additional_fusion_pattern": {
            >>>      (torch.nn.BatchNorm2d, torch.nn.Conv2d): ConvReluFusionhandler
            >>>   },

            >>>   # Additional quantization patterns
            >>>   "additional_quant_pattern": {
            >>>      torch.nn.Conv2d: ConvReluQuantizeHandler,
            >>>      (torch.nn.ReLU, torch.nn.Conv2d): ConvReluQuantizeHandler,
            >>>   }

            >>>   # By default, inputs and outputs of the graph are assumed to be in
            >>>   # fp32. Providing `input_quantized_idxs` will set the inputs with the
            >>>   # corresponding indices to be quantized. Providing
            >>>   # `output_quantized_idxs` will set the outputs with the corresponding
            >>>   # indices to be quantized.
            >>>   "input_quantized_idxs": [0],
            >>>   "output_quantized_idxs": [0],

            >>>   # Attributes that are not used in forward function will
            >>>   # be removed when constructing GraphModule, this is a list of attributes
            >>>   # to preserve as an attribute of the GraphModule even when they are
            >>>   # not used in the code, these attributes will also persist through deepcopy
            >>>   "preserved_attributes": ["preserved_attr"],
            >>> }
            ```

        equalization_qconfig_dict (`Dict[str, Any]`, *optional*):
            Refer to PyTorch documentation.
        backend_config_dict (`Dict[str, Any]`, *optional*):
            Refer to PyTorch documentation.
        input_names (`List[str]`, *optional*):
            The input names of the model, only used to trace if model is a PreTrainedModel. This is not needed if model
            is already a GraphModule.
        check (`bool`, *optional*, defaults to `True`):
            If True, a check is done to verify that the model can be traced.

    Returns:
        `torch.quantization.fx.graph_module.ObservedGraphModule`: An ObservedGraphModule ready for calibration.
    """
    if check:
        check_if_model_is_supported(model)
    tracer_cls = QuantizationTracer
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        tracer_cls = specialized_quantization_tracer_creator(get_concrete_args(model, input_names))
    orig_quantization_tracer = torch.ao.quantization.quantize_fx.QuantizationTracer
    torch.ao.quantization.quantize_fx.QuantizationTracer = tracer_cls
    graph_module = orig_prepare_fx(
        model,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
        equalization_qconfig_dict=equalization_qconfig_dict,
        backend_config_dict=backend_config_dict,
    )
    torch.ao.quantization.quantize_fx.QuantizationTracer = orig_quantization_tracer
    return graph_module


@check_if_available
def prepare_qat_fx(
    model: Union["PreTrainedModel", GraphModule],
    qconfig_dict: Any,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
    backend_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    check: bool = True,
) -> ObservedGraphModule:
    """
    Transformers models compatible version of torch.quantization.quantize_fx.prepare_qat_fx, refer to PyTorch
    documentation for more details.

    Args:
        model (`PreTrainedModel` or `torch.fx.GraphModule`):
            The model to fuse.
        qconfig_dict (`Any`):
            Refer to PyTorch documentation.
        prepare_custom_config_dict (`Dict[str, Any]`, *optional*):
            Refer to PyTorch documentation.
        backend_config_dict (`Dict[str, Any]`, *optional*):
            Refer to PyTorch documentation.
        input_names (`List[str]`, *optional*):
            The input names of the model, only used to trace if model is a PreTrainedModel. This is not needed if model
            is already a GraphModule.
        check (`bool`, *optional*, defaults to `True`):
            If True, a check is done to verify that the model can be traced.

    Returns:
        `torch.quantization.fx.graph_module.ObservedGraphModule`: An ObservedGraphModule ready for QAT.
    """
    if check:
        check_if_model_is_supported(model)
    tracer_cls = QuantizationTracer
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        tracer_cls = specialized_quantization_tracer_creator(get_concrete_args(model, input_names))
    orig_quantization_tracer = torch.ao.quantization.quantize_fx.QuantizationTracer
    torch.ao.quantization.quantize_fx.QuantizationTracer = tracer_cls
    graph_module = orig_prepare_qat_fx(
        model,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
        backend_config_dict=backend_config_dict,
    )
    torch.ao.quantization.quantize_fx.QuantizationTracer = orig_quantization_tracer
    return graph_module
