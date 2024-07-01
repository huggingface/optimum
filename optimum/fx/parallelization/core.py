from dataclasses import dataclass, field
from typing import List, Any, List, Dict, Callable
import torch
import torch.nn as nn
import torch.distributed as dist
from functools import partial

class HashableSlice:
    def __init__(self, start : int, stop : int, step : int) -> None:
        self.start = start
        self.stop = stop
        self.step = step

    def __hash__(self) -> int:
        return hash(f'{self.start},{self.stop},{self.step}')

    def __eq__(self, value: object) -> bool:
        return isinstance(value, HashableSlice) and self.start == value.start and \
            self.stop == value.stop and self.step == value.step

    def to_slice(self) -> None:
        return slice(self.start, self.stop, self.step)


@dataclass
class ParameterMeta:
    # parameter name
    source : str = None
    # which axis to index
    dim : int = None
    # index to slice the tensor
    index : slice = None


@dataclass
class ParameterMapping:
    id : int = None
    meta : ParameterMeta = None


@dataclass
class ParallelParameterMapping(ParameterMapping):
    # the axis being parallelized
    parallel_dim : int = None
    # for multi-source parameter mapping
    mapping : Dict[HashableSlice, ParameterMeta] = field(default_factory=dict)


@dataclass
class ParallelExecutionCtx:
    """
    Parallel execution context which contains runtime information.

    - example_inputs
        A list of tensors which are used as example inputs for graphs captured by dynamo.

    - parallel_layer_cache
        Cache which maps layers(`nn.Linear`, `nn.Embedding`) to their parallel counterparts.
        Note that we will build the cache in the first compilation process, and for recompilations
        later on, we will directly replace the modules with their parallel counterparts in the cache,
        because we have to make sure we don't initiate new parameters and replace original ones when
        recompilation happens in training process.

    - parameter_mapping
        Mapping between parameter ids and their correponding names in the original module. Note
        that it changes as we create new parameters to replace original ones in the first compilation
        process. It's useful because dynamo flattens the graph(which invalidates the parameter name
        hierarchy) but the original parameters are kept.

    - weight_map
        Mapping between parameter names and their locations on disk, useful when loading weights
        from disk.

    - tp_group
        Tensor parallel process group the current process belongs to.

    - compile_times
        Number of compilation times happened during the whole process.

    - current_device
        Device correpsonding to the current process.
    """
    example_inputs : List[Any] = field(default_factory=list)
    parallel_layer_cache : Dict[int, nn.Module] = field(default_factory=dict)
    parameter_mapping : Dict[int, ParameterMapping] = field(default_factory=dict)
    weight_map : Dict[str, str] = field(default_factory=dict)
    tp_group : dist.ProcessGroup = None
    compile_times : int = 0
    current_device : torch.device = None


@dataclass
class Config:
    """
    Static config which contains instructions which do not change in runtime.

    - lint_and_recompile
        Whether to run graph linting and module recompilation after every pass.

    - clean_markers_after_all_passes
        Whether to clean markers of analytical passes after all passes have run.
    
    - weight_init_fn
        Initialization function of weights in `nn.Linear` and `nn.Embedding` layers,
        if not provided weights loading path.
    """
    lint_and_recompile : bool = True
    clean_markers_after_all_passes : bool = True
    weight_init_fn : Callable = partial(nn.init.normal_, std=0.02)
