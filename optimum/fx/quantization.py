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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.fx.node import Argument, Node, Target
from torch.nn.intrinsic import _FusedModule
from torch.quantization.fx.graph_module import GraphModule, ObservedGraphModule
from torch.quantization.quantize_fx import Scope, ScopeContextManager
from torch.quantization.quantize_fx import fuse_fx as orig_fuse_fx
from torch.quantization.quantize_fx import prepare_fx as orig_prepare_fx
from torch.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx
from transformers.utils.fx import HFTracer, check_if_model_is_supported, get_concrete_args, symbolic_trace


if TYPE_CHECKING:
    from torch.fx import Graph
    from transformers import PreTrainedModel


class QuantizationTracer(HFTracer):
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

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            (m.__module__.startswith("torch.nn") and not isinstance(m, torch.nn.Sequential))
            or module_qualified_name in self.skipped_module_names
            or type(m) in self.skipped_module_classes
            or isinstance(m, _FusedModule)
            or super().is_leaf_module(m, module_qualified_name)
        )

    def call_module(
        self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        module_qualified_name = self.path_of_module(m)
        # Creating scope with information of current module
        # scope will be restored automatically upon exit
        with ScopeContextManager(self.scope, m, module_qualified_name):
            return super().call_module(m, forward, args, kwargs)

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


def specialized_quantization_tracer_creator(concrete_args):
    return type("QuantizationTracer", (QuantizationTracer,), {"specialized_concrete_args": concrete_args})


def fuse_fx(
    model: Union["PreTrainedModel", GraphModule],
    fuse_custom_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
) -> GraphModule:
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        model = symbolic_trace(model, input_names)
    orig_symbolic_trace = torch.fx.symbolic_trace
    torch.fx.symbolic_trace = lambda x: x
    gm = orig_fuse_fx(model, fuse_custom_config_dict=fuse_custom_config_dict)
    torch.fx.symbolic_trace = orig_symbolic_trace
    return gm


def prepare_fx(
    model: Union["PreTrainedModel", GraphModule],
    qconfig_dict: Any,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
    equalization_qconfig_dict: Optional[Dict[str, Any]] = None,
    backend_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
) -> ObservedGraphModule:
    check_if_model_is_supported(model)
    tracer_cls = QuantizationTracer
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        tracer_cls = specialized_quantization_tracer_creator(get_concrete_args(model, input_names))
    orig_quantization_tracer = torch.ao.quantization.quantize_fx.QuantizationTracer
    torch.ao.quantization.quantize_fx.QuantizationTracer = tracer_cls
    gm = orig_prepare_fx(
        model,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
        equalization_qconfig_dict=equalization_qconfig_dict,
        backend_config_dict=backend_config_dict,
    )
    torch.ao.quantization.quantize_fx.QuantizationTracer = orig_quantization_tracer
    return gm


def prepare_qat_fx(
    model: torch.nn.Module,
    qconfig_dict: Any,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
    backend_config_dict: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
) -> ObservedGraphModule:
    check_if_model_is_supported(model)
    tracer_cls = QuantizationTracer
    if not isinstance(model, GraphModule):
        if input_names is None:
            input_names = model.dummy_inputs.keys()
        input_names = list(input_names)
        tracer_cls = specialized_quantization_tracer_creator(get_concrete_args(model, input_names))
    orig_quantization_tracer = torch.ao.quantization.quantize_fx.QuantizationTracer
    torch.ao.quantization.quantize_fx.QuantizationTracer = tracer_cls
    gm = orig_prepare_qat_fx(
        model,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
        backend_config_dict=backend_config_dict,
    )
    torch.ao.quantization.quantize_fx.QuantizationTracer = orig_quantization_tracer
    return gm
