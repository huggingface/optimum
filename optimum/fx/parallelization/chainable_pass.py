from __future__ import annotations
from typing import Type, List, Dict, Optional, Any
from abc import ABC, abstractmethod
from torch.fx import GraphModule
from .core import ExecutionCtx, PassPipelineConfig
import warnings


class Chainable:
    def __init__(self, next : Optional[Chainable]= None) -> None:
        self._next = next

    @property
    def next(self) -> Optional[Chainable]:
        return self._next
    
    @next.setter
    def next(self, next : Optional[Chainable] = None):
        self._next = next


class PassBase(ABC):
    dependencies : List[Type[PassBase]] = []

    @property
    def signature(self) -> int:
        return id(self)

    @abstractmethod
    def run(self, graph_module : GraphModule, **kwargs: Any) -> GraphModule:
        raise NotImplementedError("Implement this first.")


class ChainablePass(Chainable, PassBase):
    def __init__(self, next: Optional[ChainablePass] = None) -> None:
        super().__init__(next)
        super(Chainable, self).__init__()

    def extract_depending_passes(
        self,
        passes : Dict[Type[ChainablePass], List[ChainablePass]]
    ) -> List[ChainablePass]:
        depending_passes = []
        for dependency_pass_type in self.dependencies:
            if dependency_pass_type not in passes:
                raise RuntimeError(
                    f"No {dependency_pass_type.__name__} in the current pipeline, please considering adding it before {self.__class__.__name__}"
                )
            elif len(passes[dependency_pass_type]) >= 2:
                warnings.warn(
                    f"Multiple {dependency_pass_type.__name__} found in current pipeline, this might incur incorrect results"
                )
            depending_passes.append(passes[dependency_pass_type][-1])
        return passes

    def __call__(
        self,
        graph_module: GraphModule,
        passes: Dict[Type[ChainablePass], List[ChainablePass]] = {},
        ctx: ExecutionCtx = None,
        lint_and_recompile: bool = True,
        clean_markers_after_all_passes: bool = True,
        **kwargs
    ) -> GraphModule:
        graph_module = self.run(graph_module, passes, ctx, **kwargs)
        if lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        if self.next:
            passes[self.__class__].append(self)
            graph_module = self.next(graph_module, passes, ctx, **kwargs)
        
        from .analyze import AnalyzeBase
        if clean_markers_after_all_passes and isinstance(self, AnalyzeBase):
            self.clean_all()
        return graph_module


def build_passes_from_config(config : PassPipelineConfig) -> List[ChainablePass]:
    # we traverse the all pass configs in dependency-aware order and collect them if they are active

    from .analyze import PostDominatorSolverPass, DependencySetSolverPass, ParallelLinearAnnotatePass
    passes = []

    if config.post_dominator_solver_config.is_active:
        passes.append(PostDominatorSolverPass(node_filter=config.post_dominator_solver_config.node_filter))
    if config.dependency_set_solver_config.is_active:
        passes.append(DependencySetSolverPass(node_filter=config.dependency_set_solver_config.node_filter))
    if config.parellel_linear_annotate_config.is_active:
        passes.append(ParallelLinearAnnotatePass())
    return passes


class ChainablePassPipeline:
    def __init__(
        self,
        passes : List[ChainablePass] = [],
        config : PassPipelineConfig = None,
    ) -> None:
        if len(passes) and config is not None:
            raise RuntimeError(
                "You can't initiate both `passes` and `config` arguments because there might be"
                " conflicts, and `ChainablePassPipeline` won't try detecting and correcting it."
            )
        if config is not None:
            passes = build_passes_from_config(config)
        
        self.lead = passes[0] if len(passes) else None
        for (prev, next) in zip(passes[:-1], passes[1:]):
            prev.next = next

    @classmethod
    def from_config(cls, config : PassPipelineConfig):
        return cls(config=config)
    
    def __call__(
        self,
        graph_module: GraphModule,
        passes: Dict[Type[ChainablePass], List[ChainablePass]] = {},
        ctx: ExecutionCtx = None,
        lint_and_recompile : bool = True,
        clean_markers_after_all_passes : bool = True,
        **kwargs: Any
    ) -> GraphModule:
        if self.lead is not None:
            graph_module = self.lead(
                graph_module, 
                passes=passes,
                ctx=ctx,
                lint_and_recompile=lint_and_recompile,
                clean_markers_after_all_passes=clean_markers_after_all_passes,
                **kwargs
            )
        return graph_module