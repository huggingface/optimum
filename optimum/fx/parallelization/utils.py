import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Node

def is_linear(node : Node) -> bool:
    if node.op != 'call_module':
        return False
    mod = node.graph.owning_module
    return isinstance(mod.get_submodule(node.target), nn.Linear)

def is_matmul(node : Node) -> bool:
    if node.op != 'call_function':
        return False
    return node.target is torch.matmul

def is_sdpa(node : Node) -> bool:
    if node.op != 'call_function':
        return False
    return node.target is torch._C._nn.scaled_dot_product_attention

def is_activation(node : Node) -> bool:
    if node.op == 'call_function':
        return node.target in {F.gelu, F.silu, F.sigmoid, F.relu, }
    elif node.op == 'call_module':
        mod = node.graph.owning_module
        return isinstance(mod.get_submodule(node.target), (nn.GELU, nn.SiLU, nn.Sigmoid, nn.ReLU))
    return False