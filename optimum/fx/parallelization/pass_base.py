from __future__ import annotations
from typing import List, Any
from abc import ABC, abstractmethod
from torch.fx import GraphModule
from .core import ExecutionCtx, PassPipelineConfig


class PassBase(ABC):
    @classmethod
    def signature(cls) -> str:
        return cls.__name__

    @abstractmethod
    def run(self, graph_module : GraphModule, **kwargs: Any) -> GraphModule:
        raise NotImplementedError("Implement this first.")

    def __call__(
        self,
        graph_module: GraphModule,
        ctx: ExecutionCtx = ExecutionCtx(),
        lint_and_recompile: bool = True,
        **kwargs
    ) -> GraphModule:
        graph_module = self.run(graph_module, ctx=ctx, **kwargs)
        if lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module


def build_passes_from_config(config : PassPipelineConfig) -> List[PassBase]:
    # we traverse the all pass configs in dependency-aware order and collect them if they are active

    from .analyze import (
        ShapePropagationPass,
        PostDominatorSolverPass,
        DependencySetSolverPass,
        ParallelLinearAnnotatePass,
        AttentionHeadIndexPropagationPass,
    )
    passes = []
    if config.shape_propagation_config.is_active:
        passes.append(ShapePropagationPass())
    if config.post_dominator_solver_config.is_active:
        passes.append(PostDominatorSolverPass(node_filter=config.post_dominator_solver_config.node_filter))
    if config.dependency_set_solver_config.is_active:
        passes.append(DependencySetSolverPass(node_filter=config.dependency_set_solver_config.node_filter))
    if config.parellel_linear_annotate_config.is_active:
        passes.append(ParallelLinearAnnotatePass())
    if config.attention_head_index_propagation_config.is_active:
        passes.append(AttentionHeadIndexPropagationPass())
    
    return passes


class PassPipeline:
    def __init__(
        self,
        passes : List[PassBase] = [],
        config : PassPipelineConfig = None,
    ) -> None:
        if len(passes) and config is not None:
            raise RuntimeError(
                "You can't initiate both `passes` and `config` arguments because there might be"
                " conflicts, and `PassPipeline` won't try detecting and correcting it."
            )
        if config is not None:
            passes = build_passes_from_config(config)
        
        self._passes = passes

    @classmethod
    def from_config(cls, config : PassPipelineConfig):
        return cls(config=config)

    def __iter__(self,):
        return self._passes.__iter__()
    
    def __call__(
        self,
        graph_module: GraphModule,
        ctx: ExecutionCtx = ExecutionCtx(),
        lint_and_recompile : bool = True,
        clean_markers_after_all_passes : bool = True,
        **kwargs: Any
    ) -> GraphModule:
        for PASS in self._passes:
            graph_module = PASS(
                graph_module=graph_module,
                ctx=ctx,
                lint_and_recompile=lint_and_recompile
            )
        
        from .analyze import AnalyzeBase

        if clean_markers_after_all_passes:
            for PASS in self._passes:
                if isinstance(PASS, AnalyzeBase):
                    PASS.clean_all(graph_module)
        return graph_module