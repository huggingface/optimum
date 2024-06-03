from typing import Any, Dict, List, Type, Callable, Optional
from torch.fx import Graph, GraphModule, Node
from torch._inductor.pattern_matcher import stable_topological_sort
from functools import reduce
from collections import defaultdict
from .chainable_pass import ChainablePass
from .utils import is_linear, is_sdpa, is_activation, is_matmul


class AnalyzeBase(ChainablePass):
    # place unique meta_key in `meta` to prevent duplicate fields
    @property
    def meta_key(self) -> str:
        return f'{self.signature()}'

    def get_stored_field_info(self, node : Node, field : Any) -> Any:
        if not self.already_executed_per_node(node):
            return None

        info : Dict[Any, Any] = node.meta[self.meta_key]
        if field not in info:
            raise ValueError(f"Invalid query field {field} for {self.__name__}, valid fields are {list(info.keys())}")

        return info[field]
    
    def already_executed_per_node(self, node : Node) -> None:
        return self.meta_key in node.meta
    
    def place_marker_per_node(self, node : Node, info : Dict[Any, Any]) -> None:
        node.meta[self.meta_key] = info

    def clear_marker_per_node(self, node : Node) -> None:
        if self.meta_key in node.meta:
            node.meta.pop(self.meta_key)

    def clean_all(self, graph_module : GraphModule) -> None:
        g : Graph = graph_module.graph
        for node in g.nodes:
            self.clear_marker_per_node(node)


class PostDominatorSolverPass(AnalyzeBase):
    def __init__(
        self,
        node_filter : Callable[[Node], bool] = lambda x : True,
        next: Optional[ChainablePass] = None) -> None:
        super().__init__(next)
        self.node_filter = node_filter

    def run(self, graph_module: GraphModule, **kwargs) -> GraphModule:
        g : Graph = graph_module.graph
        stable_topological_sort(g)

        for node in reversed(g.nodes):
            doms = {node}
            candidates = []
            for user in node.users:
                dom = self.get_stored_field_info(user, 'post_doms')
                assert dom is not None
                candidates.append(dom)
            if len(candidates):
                doms = doms.union(reduce(lambda x, y: x.intersection(y), candidates))
            self.place_marker_per_node(node, {'post_doms' : doms})

        for node in g.nodes:
            if not self.node_filter(node):
                self.clear_marker_per_node()

        return graph_module


class DependencySetSolverPass(AnalyzeBase):
    def __init__(
        self,
        node_filter : Callable[[Node], bool] = lambda x : True,
        next: Optional[ChainablePass] = None) -> None:
        super().__init__(next)
        self.node_filter = node_filter
    def run(self, graph_module: GraphModule, **kwargs) -> GraphModule:
        g : Graph = graph_module.graph
        stable_topological_sort(g)

        for node in g.nodes:
            deps = {node}
            candidates = []
            for pred in node.all_input_nodes:
                dep = self.get_stored_field_info(pred, 'dependency_nodes')
                assert dep is not None
                candidates.append(dep)
            deps = reduce(lambda x, y: x.union(y), candidates, deps)
            self.place_marker_per_node(node, {'dependency_nodes' : deps})

        for node in g.nodes:
            if not self.node_filter(node):
                self.clear_marker_per_node()

        return graph_module


class ParallelLinearAnnotatePass(AnalyzeBase):
    dependencies = [PostDominatorSolverPass, DependencySetSolverPass]

    def mark_attention_related_linears(
        self,
        graph : Graph,
        dependency_set_solver_pass : AnalyzeBase,
        post_dominator_solver_pass : AnalyzeBase,
        downstream_linears : List[Node]
    ) -> None:
        deps, post_doms = [], []
        for linear in downstream_linears:
            dep = dependency_set_solver_pass.get_stored_field_info(linear, field='dependency_nodes')
            assert dep is not None, "`DependencySetSolverPass` must have run before `ParallelLinearAnnotatePass`"
            deps.append(dep)

            post_dom = post_dominator_solver_pass.get_stored_field_info(linear, 'post_doms')
            assert post_dom is not None, "`PostDominatorSolverPass` must have run before `ParallelLinearAnnotatePass`"
            post_doms.append(post_dom)

        # Check 1: no dependencies between parallel linears
        if {downstream_linears[0], downstream_linears[1]}.intersection(deps[2]) or \
            {downstream_linears[1], downstream_linears[2]}.intersection(deps[0]) or \
            {downstream_linears[0], downstream_linears[2]}.intersection(deps[1]):
            return

        # Check 2: there is a Linear after these three Linears and it post-dominates these three linears
        # Need topo-order here
        node, last_node = downstream_linears[-1].next, next(reversed(graph.nodes))
        sdpas, matmul_2, matmul_3 = 0, 0, 0
        while node is not last_node and not is_linear(node):
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
        for linear in downstream_linears:
            self.place_marker_per_node(linear, {'replace_by' : 'column'})

        self.place_marker_per_node(node, {'replace_by' : 'row'})


    def mark_mlp_related_linears(
        self,
        graph : Graph,
        dependency_set_solver_pass : AnalyzeBase,
        post_dominator_solver_pass : AnalyzeBase,
        linears : List[Node]
    ) -> None:
        if any([self.already_executed_per_node(node) for node in linears]):
            return

        deps, post_doms = [], []
        for linear in linears:
            dep = dependency_set_solver_pass.get_stored_field_info(linear, field='dependency_nodes')
            assert dep is not None, "`DependencySetSolverPass` must have run before `ParallelLinearAnnotatePass`"
            deps.append(dep)

            post_dom = post_dominator_solver_pass.get_stored_field_info(linear, 'post_doms')
            assert post_dom is not None, "`PostDominatorSolverPass` must have run before `ParallelLinearAnnotatePass`"
            post_doms.append(post_dom)

        if len(linears) == 2 and linears[0] in deps[1] or linears[1] in deps[0]:
            return

        node, last_node = linears[-1].next, next(reversed(graph.nodes))

        activations = 0
        while node is not last_node and not is_linear(node):
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


    def run(
        self,
        graph_module: GraphModule,
        passes : Dict[Type[ChainablePass], ChainablePass],
        **kwargs
    ) -> GraphModule:
        g : Graph = graph_module.graph
        stable_topological_sort(g)
        
        linear_groups : Dict[Node, List[Node]] = defaultdict(list)
        for node in g.nodes:
            if is_linear(node):
                linear_groups[node.args[0]].append(node)

        dependency_set_solver_pass, post_dominator_solver_pass  = self.extract_depending_passes(passes)

        # first process attention-related linears, q_proj, k_proj, v_proj, o_proj
        for _, downstream_linears in linear_groups.items():
            if len(downstream_linears) == 3:
                self.mark_attention_related_linears(g, dependency_set_solver_pass, post_dominator_solver_pass, downstream_linears)
        
        # then llama-style mlp
        for _, downstream_linears in linear_groups.items():
            if len(downstream_linears) == 2:
                self.mark_mlp_related_linears(g, dependency_set_solver_pass, post_dominator_solver_pass, downstream_linears)
        
        # finally classic-style mlp
        for _, downstream_linears in linear_groups.items():
            if len(downstream_linears) == 1:
                self.mark_mlp_related_linears(g, dependency_set_solver_pass, post_dominator_solver_pass, downstream_linears)

        return graph_module