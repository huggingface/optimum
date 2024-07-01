import torch
from torch.fx import GraphModule
from typing import List
from .core import ParallelExecutionCtx, Config
from .passes import build_parallel_pass_pipeline


def parallelize_backend(graph_module: GraphModule, example_inputs: List[torch.Tensor], ctx: ParallelExecutionCtx, config: Config):
    ctx.example_inputs = example_inputs
    pass_pipeline = build_parallel_pass_pipeline()
    graph_module = pass_pipeline(graph_module=graph_module, ctx=ctx, config=config)
    ctx.compile_times += 1
    return graph_module
