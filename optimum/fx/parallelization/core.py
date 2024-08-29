# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import GraphModule


class HashableSlice:
    def __init__(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> None:
        self.start = start
        self.stop = stop
        self.step = step

    def __hash__(self) -> int:
        return hash(f"{self.start},{self.stop},{self.step}")

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, HashableSlice)
            and self.start == value.start
            and self.stop == value.stop
            and self.step == value.step
        )

    def to_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)


@dataclass
class ParameterSlice:
    """
    A slice of parameter which corresponds to a tensor in weight dict. Only support slicing
    along a specific axis (the potential parallel axis) right now.

    Attributes:
        - source (`Optional[str]`, defaults to `None`):
            Original parameter name which can be found in the weight dict.

        - shape (`Optional[Tuple]`, defaults to `None`):
            Shape of parameter tensor corresponding to `source`.

        - index (`slice`, defaults to `slice(None, None, None)`):
            Index to slice the tensor on the parallel axis. Assume tensor in weight dict has the same
            layout as their correspondings in memory.
    """

    source: Optional[str] = None
    shape: Optional[Tuple] = None
    index: slice = slice(None, None, None)


@dataclass
class ParameterMeta:
    """
    Parameter meta information.

    Attributes:
        - is_tied (`bool`, defaults to `False`):
            Whether the parameter is shared accross multiple modules.

        - is_parallel (`bool`, defaults to `False`):
            Whether the parameter needs to be parallelized.

        - is_modified_meta (`bool`, defaults to `False`):
            Whether the meta has already been modified since initialization.

        - need_initialize (`bool`, defaults to `False`):
            Whether need to manually initialize weights if not provided in weight map.

        - init_fn (`Optional[Callable]`, defaults to `None`):
            Initialization function, can override `weight_init_fn` in `Config` if not None.

        - dim (`int`, defaults to `0`):
            Axis on which `mapping` is based, also the parallel axis if `is_parallel`.

        - mapping (`Dict[HashableSlice, ParameterSlice]`):
            Mapping between the current parameter and weight tensor stored in weight map.
    """

    is_tied: bool = False
    is_parallel: bool = False
    is_modified_meta: bool = False
    need_initialize: bool = False
    init_fn: Optional[Callable] = None
    dim: int = 0
    mapping: Dict[HashableSlice, ParameterSlice] = field(default_factory=dict)


@dataclass
class ParallelExecutionCtx:
    """
    Parallel execution context which contains runtime information.

    Attributes:
        - tp_group (`dist.ProcessGroup`):
            Tensor parallel process group the current process belongs to.

        - current_device (`torch.device`):
            Device correpsonding to the current process.

        - example_inputs (`List[Any]`):
            A list of tensors which are used as example inputs for graphs captured by dynamo.

        - parallel_layer_cache (`Dict[str, nn.Module]`):
            Cache which maps layers(`nn.Linear`, `nn.Embedding`) to their parallel counterparts.
            Note that we will build the cache in the first compilation process, and for recompilations
            later on, we will directly replace the modules with their parallel counterparts in the cache,
            because we have to make sure we don't initiate new parameters and replace original ones when
            recompilation happens in training process.

        - param_cache (`Dict[str, nn.Parameter]`):
            Cache which keeps record of newly created parameters. Similar to `parallel_layer_cache`, we
            need to make sure all the newly created parameters in the first compilation will still be used
            when recompilation happens.

        - weight_map (`Dict[str, str]`):
            Mapping between parameter names and their locations on disk, useful when loading weights
            from disk.

        - last_optimized_graph_module (`Optional[GraphModule]`, defaults to `None`):
            Optimized graph module corresponding to the latest compilation.

        - compile_times (`int`, defaults to `0`):
            Number of compilation times happened during the whole process.
    """

    tp_group: dist.ProcessGroup
    current_device: torch.device
    example_inputs: List[Any] = field(default_factory=list)
    parallel_layer_cache: Dict[str, nn.Module] = field(default_factory=dict)
    param_cache: Dict[str, nn.Parameter] = field(default_factory=dict)
    weight_map: Dict[str, str] = field(default_factory=dict)
    last_optimized_graph_module: Optional[GraphModule] = None
    compile_times: int = 0


@dataclass
class Config:
    """
    Static config which contains instructions which do not change in runtime.

    Attributes:
        - lint_and_recompile (`bool`, defaults to `True`):
            Whether to run graph linting and module recompilation after every pass.

        - clean_markers_after_all_passes (`bool`, defaults to `True`):
            Whether to clean markers of analytical passes after all passes have run.

        - weight_init_fn (`Callable`, defaults to `partial(nn.init.normal_, std=0.02)`)
            Initialization function of weights in `nn.Linear` and `nn.Embedding` layers,
            if not provided weights loading path.

        - enable_sequence_parallel (`bool`, defaults to `False`):
            Whether to enable Megatron-style sequence parallelism in searching parallelization
            strategies.
    """

    lint_and_recompile: bool = True
    clean_markers_after_all_passes: bool = True
    weight_init_fn: Callable = partial(nn.init.normal_, std=0.02)
    enable_sequence_parallel: bool = False
