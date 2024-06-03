import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable
from torch.fx import Node


PARALLEL_INTERESTED_NODES = (
    ('call_module', nn.Linear),
    ('call_module', nn.GELU),
    ('call_module', nn.SiLU),
    ('call_function', torch.matmul),
    ('call_function', F.scaled_dot_product_attention),
    ('call_function', F.gelu),
    ('call_function', F.silu),
)

@dataclass
class PassConfig:
    is_active : bool = False

@dataclass
class PostDominatorSolverConfig(PassConfig):
    # only information of nodes satisfying `node_filter` will be kept
    # for later uses in consideration of memory consumption
    node_filter : Callable[[Node], bool] = lambda x : True

@dataclass
class DependencySetSolverConfig(PassConfig):
    # only information of nodes satisfying `node_filter` will be kept
    # for later uses in consideration of memory consumption
    node_filter : Callable[[Node], bool] = lambda x : True

@dataclass
class ParallelLinearAnnotateConfig(PassConfig):
    pass

@dataclass
class PassPipelineConfig:
    post_dominator_solver_config : PostDominatorSolverConfig = PostDominatorSolverConfig()
    dependency_set_solver_config : DependencySetSolverConfig = DependencySetSolverConfig()
    parellel_linear_annotate_config : ParallelLinearAnnotateConfig = ParallelLinearAnnotateConfig()