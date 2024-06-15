from typing import Any, Dict, List, Callable
from torch.fx import Graph, GraphModule, Node
from torch._inductor.pattern_matcher import stable_topological_sort
from torch.fx.passes.shape_prop import ShapeProp
from functools import reduce
from collections import defaultdict
from .pass_base import PassBase
from .utils import (
    is_linear,
    is_sdpa,
    is_activation,
    is_matmul,
    is_transpose,
    is_permute,
    is_getitem,
)
from .core import ExecutionCtx


class AnalyzeBase(PassBase):
    # place class-wise unique meta_key in `meta` to prevent duplicate fields
    @classmethod
    def meta_key(cls) -> str:
        return cls.signature()
    
    @classmethod
    def get_stored_field_info(cls, node : Node, field : Any, must_have : bool = False) -> Any:
        if not cls.already_executed_per_node(node):
            if not must_have:
                return None
            else:
                raise RuntimeError(
                    f"Can't find information related with {cls.__name__} in the current node `{node}`"
                    "make sure {cls.__name__} has run and marked it"
                )
        
        info : Dict[Any, Any] = node.meta[cls.meta_key()]
        if field not in info:
            raise KeyError(f"Invalid query field {field} for {cls.__name__}, valid fields are {list(info.keys())}")

        return info[field]
    
    @classmethod
    def already_executed_per_node(cls, node : Node) -> None:
        return cls.meta_key() in node.meta
    
    def place_marker_per_node(self, node : Node, info : Dict[Any, Any]) -> None:
        if self.already_executed_per_node(node):
            raise RuntimeError(
                f"Node {node} has already been marked by the current pass, check if "
                "the current pass has already been executed in the pipeline"
            )

        node.meta[self.meta_key()] = info

    def clear_marker_per_node(self, node : Node) -> None:
        key = self.meta_key()
        if key in node.meta:
            node.meta.pop(key)

    def clean_all(self, graph_module : GraphModule) -> None:
        g : Graph = graph_module.graph
        for node in g.nodes:
            self.clear_marker_per_node(node)


class ShapePropagationPass(AnalyzeBase):
    def run(self, graph_module: GraphModule, ctx: ExecutionCtx, **kwargs) -> GraphModule:
        example_inputs = ctx.example_inputs
        ShapeProp(graph_module).propagate(*example_inputs)
        return graph_module


class PostDominatorSolverPass(AnalyzeBase):
    def __init__(self, node_filter : Callable[[Node], bool] = lambda x : True) -> None:
        super().__init__()
        self.node_filter = node_filter

    def run(self, graph_module: GraphModule, **kwargs) -> GraphModule:
        g : Graph = graph_module.graph
        stable_topological_sort(g)

        for node in reversed(g.nodes):
            doms = {node}
            candidates = []
            for user in node.users:
                dom = self.get_stored_field_info(user, field='post_doms', must_have=True)
                candidates.append(dom)
            if len(candidates):
                doms = doms.union(reduce(lambda x, y: x.intersection(y), candidates))
            self.place_marker_per_node(node, {'post_doms' : doms})

        for node in g.nodes:
            if not self.node_filter(node):
                self.clear_marker_per_node(node)

        return graph_module


class DependencySetSolverPass(AnalyzeBase):
    def __init__(self, node_filter : Callable[[Node], bool] = lambda x : True) -> None:
        super().__init__()
        self.node_filter = node_filter
    def run(self, graph_module: GraphModule, **kwargs) -> GraphModule:
        g : Graph = graph_module.graph
        stable_topological_sort(g)

        for node in g.nodes:
            deps = {node}
            candidates = []
            for pred in node.all_input_nodes:
                dep = self.get_stored_field_info(pred, field='dependency_nodes', must_have=True)
                candidates.append(dep)
            deps = reduce(lambda x, y: x.union(y), candidates, deps)
            self.place_marker_per_node(node, {'dependency_nodes' : deps})

        for node in g.nodes:
            if not self.node_filter(node):
                self.clear_marker_per_node(node)

        return graph_module


class ParallelLinearAnnotatePass(AnalyzeBase):
    def mark_attention_related_linears(self, graph : Graph, linears : List[Node]) -> None:
        deps, post_doms = [], []
        for linear in linears:
            dep = DependencySetSolverPass.get_stored_field_info(linear, field='dependency_nodes', must_have=True)
            deps.append(dep)

            post_dom = PostDominatorSolverPass.get_stored_field_info(linear, field='post_doms', must_have=True)
            post_doms.append(post_dom)

        # Check 1: no dependencies between parallel linears
        if {linears[0], linears[1]}.intersection(deps[2]) or \
            {linears[1], linears[2]}.intersection(deps[0]) or \
            {linears[0], linears[2]}.intersection(deps[1]):
            return

        # Check 2: there is a Linear after these three Linears and it post-dominates these three linears
        # Need topo-order here
        node, last_node = linears[0].next, next(iter(reversed(graph.nodes)))
        sdpas, matmul_2, matmul_3 = 0, 0, 0
        while node is not last_node and (node in linears or not is_linear(node)):
            if is_matmul(node):
                doms = sum([int(node in post_dom) for post_dom in post_doms])
                if doms == 2:
                    # we find a matmul dominating the two linears(Q,K) out of all three linears
                    matmul_2 += 1
                elif doms == 3 and matmul_2 == 1:
                    # we find a matmul dominating the previous matmul and all three linears
                    matmul_3 += 1
            elif is_sdpa(node) and all([node in post_dom for post_dom in post_doms]):
                sdpas += 1
            node = node.next

        if node is last_node or any([node not in post_dom for post_dom in post_doms]):
            return

        # Check 3: there is two dominating matmuls or there is one dominating sdpa
        if not ((sdpas == 1) ^ (matmul_2 == 1 and matmul_3 == 1)):
            return

        # we can almost certainly say we have captured an self-attention pattern here,
        # we will be fine as long as we are right under 99% of situations
        for linear in linears:
            self.place_marker_per_node(linear, {'replace_by' : 'column'})

        self.place_marker_per_node(node, {'replace_by' : 'row'})


    def mark_mlp_related_linears(self, graph : Graph, linears : List[Node]) -> None:
        if any([self.already_executed_per_node(node) for node in linears]):
            return

        deps, post_doms = [], []
        for linear in linears:
            dep = DependencySetSolverPass.get_stored_field_info(linear, field='dependency_nodes', must_have=True)
            deps.append(dep)

            post_dom = PostDominatorSolverPass.get_stored_field_info(linear, field='post_doms', must_have=True)
            post_doms.append(post_dom)

        if len(linears) == 2 and (linears[0] in deps[1] or linears[1] in deps[0]):
            return

        node, last_node = linears[0], next(iter(reversed(graph.nodes)))

        activations = 0
        while node is not last_node and (node in linears or not is_linear(node)):
            if is_activation(node) and sum([int(node in post_dom) for post_dom in post_doms]):
                activations += 1
            node = node.next

        if node is last_node or self.already_executed_per_node(node) or any([node not in post_dom for post_dom in post_doms]):
            return
        
        # should have at least one activation node in between
        if activations == 0:
            return

        for linear in linears:
            self.place_marker_per_node(linear, {'replace_by' : 'column'})

        self.place_marker_per_node(node, {'replace_by' : 'row'})


    def run(self, graph_module: GraphModule, **kwargs) -> GraphModule:
        g : Graph = graph_module.graph
        stable_topological_sort(g)
        
        linear_groups : Dict[Node, List[Node]] = defaultdict(list)
        for node in g.nodes:
            if is_linear(node):
                linear_groups[node.args[0]].append(node)

        # first process attention-related linears, q_proj, k_proj, v_proj, o_proj
        for _, downstream_linears in linear_groups.items():
            if len(downstream_linears) == 3:
                self.mark_attention_related_linears(g, downstream_linears)
        
        # then llama-style mlp
        for _, downstream_linears in linear_groups.items():
            if len(downstream_linears) == 2:
                self.mark_mlp_related_linears(g, downstream_linears)
        
        # finally classic-style mlp
        for _, downstream_linears in linear_groups.items():
            if len(downstream_linears) == 1:
                self.mark_mlp_related_linears(g, downstream_linears)

        return graph_module


class AttentionHeadIndexPropagationPass(AnalyzeBase):
    def propagate_transpose(self, node: Node, head_idx: int) -> bool:
        if 'dim0' in node.kwargs and 'dim1' in node.kwargs:
            dim0, dim1, dims = node.kwargs['dim0'], node.kwargs['dim1'], len(node.meta['tensor_meta'].shape)
            dim0 = (dim0 + dims) % dims
            dim1 = (dim1 + dims) % dims
            if dim0 == head_idx:
                self.place_marker_per_node(node, {'head_idx' : dim1})
                return True
            elif dim1 == head_idx:
                self.place_marker_per_node(node, {'head_idx' : dim0})
                return True
            return False

        if len(node.args) == 3:
            dims = len(node.meta['tensor_meta'].shape)
            if head_idx not in node.args and head_idx - dims not in node.args:
                return False
            for arg in node.args:
                if isinstance(arg, int) and (arg + dims) % dims != head_idx:
                    self.place_marker_per_node(node, {'head_idx' : (arg + dims) % dims})
                    return True
        
        return False
    
    def propagate_permute(self, node: Node, head_idx: int) -> bool:
        if 'dims' in node.kwargs:
            dims = node.kwargs['dims']
        else:
            dims = list(node.args[1]) if isinstance(node.args[1], tuple) else [arg for arg in node.args if isinstance(arg,int)]
        
        dim_len = len(node.meta['tensor_meta'].shape)
        dims = [dim + dim_len if dim < 0 else dim for dim in dims]

        for i,dim in enumerate(dims):
            if dim == head_idx:
                self.place_marker_per_node(node, {'head_idx' : i})
                return True
        return False
    
    def propagate_getitem(self, node: Node, head_idx: int) -> bool:
        slices = node.args[1]
        dims = len(node.meta['tensor_meta'].shape)
        assert head_idx < dims
        inc, i, j = 0, 0, 0

        while i < head_idx and j < len(slices):
            if isinstance(slices[j], int):
                inc -= 1
                i += 1
            elif slices[j] is None:
                inc += 1
            elif slices[j] is Ellipsis:
                i = dims
                k = j
                while k < len(slices):
                    if isinstance(slices[k], (slice, int)):
                        i -= 1
                    k += 1
            else:
                i += 1
            j += 1

        if inc != 0:
            assert head_idx + inc < dims and head_idx + inc >= 0
            self.place_marker_per_node(node, {'head_idx' : head_idx + inc})
            return True
        return False

    def run(self, graph_module: GraphModule, ctx: ExecutionCtx, **kwargs) -> GraphModule:
        g: Graph = graph_module.graph
        stable_topological_sort(g)

        for node in g.nodes:
            if ParallelLinearAnnotatePass.already_executed_per_node(node):
                # start propagating at ColumnLinear
                replace_by = ParallelLinearAnnotatePass.get_stored_field_info(node, field='replace_by', must_have=True)
                if replace_by == 'column':
                    self.place_marker_per_node(node, {'head_idx' : 2})
                # stop propagating at RowLinear, concluding the life cycle of attention heads
                else:
                    continue
            else:
                already_marked_args, head_idx = [], None
                for arg in node.all_input_nodes:
                    if not self.already_executed_per_node(arg):
                        continue
                    if head_idx is None:
                        head_idx = self.get_stored_field_info(arg, field='head_idx', must_have=True)
                    else:
                        assert head_idx == self.get_stored_field_info(arg, field='head_idx', must_have=True), \
                            "`head_idx` should be equal for all arguments in any related ops"
                    already_marked_args.append(arg)
                
                if not already_marked_args:
                    continue

                marked = False
                if is_transpose(node):
                    marked = self.propagate_transpose(node, head_idx)
                elif is_permute(node):
                    marked = self.propagate_permute(node, head_idx)
                elif is_getitem(node):
                    marked = self.propagate_getitem(node, head_idx)
                
                # fall back
                if not marked:
                    self.place_marker_per_node(node, {'head_idx' : head_idx})
        return graph_module