from __future__ import annotations
from typing import List, Any, Dict
from abc import ABC, abstractmethod
from torch.fx import GraphModule, Graph, Node
import torch.nn as nn
from .utils import (
    stable_topological_sort,
    is_transpose,
    is_permute,
    is_linear,
    is_shape_consumer,
    is_shape_generator,
)

from .core import ParallelExecutionCtx, Config
from .parallel_layers import ColumnParallelLinear, RowParallelLinear


class PassBase(ABC):
    """
    Base class for parallelization targeted passes
    """
    @classmethod
    def signature(cls) -> str:
        return cls.__name__

    @abstractmethod
    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        """
        Args:
            graph_module (`GraphModule`):
                graph module before processing.
            ctx (`ParallelExecutionCtx`):
                dynamic execution context which gathers and preserves information along processing.
            config (`Config`):
                static config to include instructions which persists the whole process.

        Returns:
            GraphModule: graph module after processed by the current pass.
        """
        raise NotImplementedError

    def __call__(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        graph_module = self.run(graph_module, ctx=ctx, config=config)
        if config.lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module


class AnalyzeBase(PassBase):
    """
    Base class for passes which only run for analytical purposes and preserve graph structure
    during processing. Analytical passes are often prerequisite passes which provide information
    for passes later on to actually change the graph.

    Passes inheriting from `AnalyBase` places the class signature as a meta key in `node.meta`,
    which is a dict storing meta information related with a fx Node, such as the shape and dtype of
    output. Look-up APIs are exposed as classmethod so that passes using them won't need to create
    concrete instances.
    """
    @classmethod
    def meta_key(cls) -> str:
        # place class-wise unique meta_key in `meta` to prevent duplicate fields
        return cls.signature()
    
    @classmethod
    def get_stored_field_info(cls, node: Node, field: Any, must_have: bool = False) -> Any:
        if not cls.already_executed_per_node(node):
            if not must_have:
                return None
            else:
                raise RuntimeError(
                    f"Can't find information related with {cls.__name__} in the current node `{node}` "
                    f"make sure {cls.__name__} has run and marked it"
                )
        
        info : Dict[Any, Any] = node.meta[cls.meta_key()]
        if field not in info:
            raise KeyError(f"Invalid query field {field} for {cls.__name__}, valid fields are {list(info.keys())}")

        return info[field]
    
    @classmethod
    def already_executed_per_node(cls, node: Node) -> None:
        return cls.meta_key() in node.meta
    
    def place_marker_per_node(self, node: Node, info: Dict[Any, Any]) -> None:
        if self.already_executed_per_node(node):
            raise RuntimeError(
                f"Node {node} has already been marked by the current pass, check if "
                "the current pass has already been executed in the pipeline"
            )

        node.meta[self.meta_key()] = info

    def clear_marker_per_node(self, node: Node) -> None:
        key = self.meta_key()
        if key in node.meta:
            node.meta.pop(key)

    def clean_all(self, graph_module: GraphModule) -> None:
        g : Graph = graph_module.graph
        for node in g.nodes:
            self.clear_marker_per_node(node)


class ParallelLinearAnnotatePass(AnalyzeBase):
    """
    A pass which tries to automatically identify parallel linears in the graph by grouping linears as
    `upstream` nodes and `downstream` nodes, and `upstream` nodes are marked as `ColumnLinear`, `downstream`
    nodes are marked as `RowLinear`.
    
    Typical examples in transformer models:

          Attention                   Bert-style MLP          Llama-style MLP
        __________________________________________________________________________
        Linear  Linear                     Linear           Linear
          \\     /                            |               \\                       --> upstream
            Matmul   Linear               Activation         Activation  Linear
        __________________________________________________________________________
               \\    /                        |                    \\     /
                \\  /                     ___________               \\   / 
                Matmul                   /  Linear   \                Mul
                   |                    /             \                |
        _______________________________/               \___________________________
                 Linear                                              Linear            --> downstream

    Note that there are some patterns that can not be clearly marked, like this one:

    Linear
      |    \\  
      |    Linear  <-- which label should we mark for the intermediate linear, `upstream` or `downstream`
      |     /
        Add
         |
       Linear
    
    For patterns like this we will be preservative and raise errors directly because we don't know how to parallelize
    it. Another concern is about the correctness, it's possible that we might end up with a wrong parallelization solution
    even if the pattern itself is clear, but for now we are mainly targeting on transformer models and the current solution
    should work fairly well.
    """
    def try_form_parallel_linear_groups(self, linear: Node) -> None:
        """
        We try to form linears by forming closures in a greedy way, we start with an unmarked linear node, and traverses down
        recusively to find all the potential `downstream` linears, note that once we have reached a linear, the recursion stops.
        And the newly found `downstream` linears are used as new seeds to traverse upwards to find all the potential `upstream`
        linears, the process goes on until number of linears on both sides converges.
        Args:
            linear (Node): the first linear node used as `upstream` node seed to form closure.

        Raises:
            RuntimeError:
                raises runtime error when the pattern itself is not clear, there are no clear boundaries that can be drawn.
        """
        upstream_nodes, downstream_nodes = {linear}, set()

        seeds, next_seeds = [(linear, 'down')], []

        def traverse(start: Node, cur: Node, direction = 'down'):
            if is_linear(cur) and cur is not start:
                if direction == 'up' and cur not in upstream_nodes:
                    upstream_nodes.add(cur)
                    next_seeds.append((cur, 'down'))
                elif direction == 'down' and cur not in downstream_nodes:
                    downstream_nodes.add(cur)
                    next_seeds.append((cur, 'up'))
                return
                

            next_nodes = cur.all_input_nodes if direction == 'up' else cur.users
            for node in next_nodes:
                # we should ignore shape-related dependencies
                if is_shape_generator(node):
                    continue
                traverse(start, node, direction)
        
        while seeds:
            next_seeds = []
            for node, direction in seeds:
                traverse(start=node, cur=node, direction=direction)
            seeds = next_seeds

        if any([self.already_executed_per_node(node) for node in (upstream_nodes | downstream_nodes)]) or \
            (upstream_nodes & downstream_nodes):
            raise RuntimeError(
                "Failed to automatically group and parallelize ops in graph in greedy way: "
                "no clear boudaries between `upstream` and `downstream` ops." 
            )
        
        for node in upstream_nodes:
            self.place_marker_per_node(node, {'axis' : 'column', 'gather_output' : False if downstream_nodes else True})

        for node in downstream_nodes:
            self.place_marker_per_node(node, {'axis' : 'row', 'input_is_parallel' : True})


    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        graph: Graph = graph_module.graph
        stable_topological_sort(graph)
        for node in graph.nodes:
            if is_linear(node) and not self.already_executed_per_node(node):
                self.try_form_parallel_linear_groups(node)
            
        return graph_module


class ParallelAxisPropagationPass(AnalyzeBase):
    """
    A pass tries to track which axis is being parallelized in the dataflow. For transformer models, the
    axis being paralled for tensor parallism is almost always 2, i.e., the attention head axis, except for
    Q and K matrice which need to swap the sequence length axis and head axis to do the attention computation,
    so we focus on operations like `transpose` or `permute` which swaps axis, and try inducting the parallel
    axis after these operations.
    """
    def propagate_transpose(self, node: Node, parallel_axis: int) -> bool:
        dims = node.meta['example_value'].dim()
        if 'dim0' in node.kwargs and 'dim1' in node.kwargs:
            dim0, dim1, dims = node.kwargs['dim0'], node.kwargs['dim1']
            dim0 = (dim0 + dims) % dims
            dim1 = (dim1 + dims) % dims
            if dim0 == parallel_axis:
                self.place_marker_per_node(node, {'parallel_axis' : dim1})
                return True
            elif dim1 == parallel_axis:
                self.place_marker_per_node(node, {'parallel_axis' : dim0})
                return True
            return False

        if len(node.args) == 3:
            if parallel_axis not in node.args and parallel_axis - dims not in node.args:
                return False
            for arg in node.args:
                if isinstance(arg, int) and (arg + dims) % dims != parallel_axis:
                    self.place_marker_per_node(node, {'parallel_axis' : (arg + dims) % dims})
                    return True
        
        return False

    def propagate_permute(self, node: Node, parallel_axis: int) -> bool:
        if 'dims' in node.kwargs:
            dims = node.kwargs['dims']
        else:
            dims = list(node.args[1]) if isinstance(node.args[1], tuple) else [arg for arg in node.args if isinstance(arg,int)]
        
        dim_len = node.meta['example_value'].dim()
        dims = [dim + dim_len if dim < 0 else dim for dim in dims]

        for i,dim in enumerate(dims):
            if dim == parallel_axis:
                self.place_marker_per_node(node, {'parallel_axis' : i})
                return True
        return False

    def propagate_getitem(self, node: Node, parallel_axis: int) -> bool:
        slices = node.args[1]
        dims = node.meta['example_value'].dim()
        assert parallel_axis < dims
        inc, i, j = 0, 0, 0

        while i < parallel_axis and j < len(slices):
            if isinstance(slices[j], int):
                inc -= 1
                i += 1
            elif slices[j] is None:
                inc += 1
            elif slices[j] is Ellipsis:
                i = dims
                k = j
                while k < len(slices):
                    if slices[k] is not Ellipsis:
                        i -= 1
                    k += 1
            else:
                i += 1
            j += 1

        if inc != 0:
            assert parallel_axis + inc < dims and parallel_axis + inc >= 0
            self.place_marker_per_node(node, {'parallel_axis' : parallel_axis + inc})
            return True
        return False

    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        g: Graph = graph_module.graph
        stable_topological_sort(g)

        for node in g.nodes:
            if ParallelLinearAnnotatePass.already_executed_per_node(node):
                # start propagating at ColumnLinear, marking the beginning of parallelized region
                axis = ParallelLinearAnnotatePass.get_stored_field_info(node, field='axis', must_have=True)
                if axis == 'column':
                    self.place_marker_per_node(node, {'parallel_axis' : 2})
                # stop propagating at RowLinear, concluding the ending of parallelized region
                else:
                    continue
            else:
                already_marked_args, parallel_axis = [], None
                for arg in node.all_input_nodes:
                    if not self.already_executed_per_node(arg):
                        continue
                    if parallel_axis is None:
                        parallel_axis = self.get_stored_field_info(arg, field='parallel_axis', must_have=True)
                    else:
                        assert parallel_axis == self.get_stored_field_info(arg, field='parallel_axis', must_have=True), \
                            "`parallel_axis` should be equal for all arguments in any related ops"
                    already_marked_args.append(arg)
                
                if not already_marked_args:
                    continue

                marked = False
                if is_transpose(node):
                    marked = self.propagate_transpose(node, parallel_axis)
                elif is_permute(node):
                    marked = self.propagate_permute(node, parallel_axis)
                
                # fall back
                if not marked:
                    self.place_marker_per_node(node, {'parallel_axis' : parallel_axis})
        return graph_module


class ParallelLinearReplacePass(PassBase):
    """
    A pass which modifies graph according to information provided by previous analytical passes,
    in general it does two things for now:
        1. replace linears with their parallel counterparts.
        2. modify hard-coded arguments like the number of attenton heads in the graph by dividing it by parallelism level.
    """
    @staticmethod
    def handle_linear(node: Node, ctx: ParallelExecutionCtx, config: Config) -> None:
        graph_module = node.graph.owning_module
        axis = ParallelLinearAnnotatePass.get_stored_field_info(node, field='axis')
        if axis is None:
            return
        
        assert axis in {'column', 'row'}
        prefix_and_field = node.target.rsplit(".", maxsplit=1)
        if len(prefix_and_field) == 2:
            parent_mod = graph_module.get_submodule(prefix_and_field[0])
            field = prefix_and_field[1]
        else:
            parent_mod = graph_module
            field = node.target

        mod : nn.Linear = graph_module.get_submodule(node.target)
        key, layer_cache = id(mod), ctx.parallel_layer_cache
        if key in layer_cache:
            new_mod = layer_cache[key]
        else:
            if axis == 'column':
                gather_output = ParallelLinearAnnotatePass.get_stored_field_info(node, field='gather_output', must_have=True)
                new_mod = ColumnParallelLinear(ctx, mod, gather_output,  config.weight_init_fn)
            else:
                input_is_parallel = ParallelLinearAnnotatePass.get_stored_field_info(node, field='input_is_parallel', must_have=True)
                new_mod = RowParallelLinear(ctx, mod, input_is_parallel, config.weight_init_fn)
            layer_cache[key] = new_mod
        setattr(parent_mod, field, new_mod)


    @staticmethod
    def handle_hard_coded_axis_param(node: Node, ctx: ParallelExecutionCtx) -> None:

        def extract_shape_from_node(node: Node) -> List[Any]:
            if 'size' in node.kwargs:
                return list(node.kwargs['size'])
            elif 'shape' in node.kwargs:
                return list(node.kwargs['shape'])
            elif isinstance(node.args[1], tuple):
                return [idx for idx in node.args[1]]
            else:
                return [idx for idx in node.args[1:]]

        def update(node: Node, new_shape: List[Any], parallel_axis: int):
            if 'size' in node.kwargs:
                node.update_kwarg('size', tuple(new_shape))
            elif 'shape' in node.kwargs:
                node.update_kwarg('shape', tuple(new_shape))
            elif isinstance(node.args[1], tuple):
                node.update_arg(1, tuple(new_shape))
            else:
                node.update_arg(parallel_axis + 1, shape[parallel_axis])

        parallel_axis = ParallelAxisPropagationPass.get_stored_field_info(node, field='parallel_axis')
        if parallel_axis is None:
            return

        shape = extract_shape_from_node(node)
        assert parallel_axis < len(shape)
        if not isinstance(shape[parallel_axis], int) or shape[parallel_axis] == -1:
            return
        world_size = ctx.tp_group.size()
        assert shape[parallel_axis] % world_size == 0
        shape[parallel_axis] = shape[parallel_axis] // world_size
        update(node, shape, parallel_axis)

    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        for node in graph_module.graph.nodes:
            if is_linear(node):
                self.handle_linear(node, ctx, config)
            # correct the attention head num in parallel setting
            elif is_shape_consumer(node):
                self.handle_hard_coded_axis_param(node, ctx)
        return graph_module


def build_parallel_pass_pipeline() -> PassPipeline:
    """
    Ensemble a pass pipeline which contains the following passes:
        
        1. `ParallelLinearAnnotatePass` to annoate which linears are `ColumnLinear`, which are `RowLinear`
        2. `ParallelAxisPropagationPass` to propate parallel axis along the data flow
        3. `ParallelLinearReplacePass` to do the actual replacement and modification of hard-coded attributes

    Returns:
        PassPipeline: the pipeline used for automatic parallelism.
    """
    return PassPipeline([
        ParallelLinearAnnotatePass(),
        ParallelAxisPropagationPass(),
        ParallelLinearReplacePass()
    ])


class PassPipeline:
    """
    `PassPipeline` ensembles a list of passes and execute them one by one as provided in the list,
    it can be iterated and appended after initialization for flexibility.
    """
    def __init__(self, passes : List[PassBase] = []) -> None:
        self._passes = passes

    def __iter__(self,):
        return self._passes.__iter__()

    def append(self, PASS: PassBase):
        self._passes.append(PASS)
    
    def __call__(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        for PASS in self._passes:
            graph_module = PASS(graph_module=graph_module, ctx=ctx, config=config)
        
        if config.clean_markers_after_all_passes:
            for PASS in self._passes:
                if isinstance(PASS, AnalyzeBase):
                    PASS.clean_all(graph_module)
        return graph_module
