import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Node
import operator

def is_linear(node: Node) -> bool:
    if node.op != 'call_module':
        return False
    mod = node.graph.owning_module
    return isinstance(mod.get_submodule(node.target), nn.Linear)

def is_matmul(node: Node) -> bool:
    if node.op != 'call_function':
        return False
    return node.target is torch.matmul

def is_sdpa(node: Node) -> bool:
    if node.op != 'call_function':
        return False
    return node.target is torch._C._nn.scaled_dot_product_attention

def is_activation(node: Node) -> bool:
    if node.op == 'call_function':
        return node.target in {F.gelu, F.silu, F.sigmoid, F.relu, }
    elif node.op == 'call_module':
        mod = node.graph.owning_module
        return isinstance(mod.get_submodule(node.target), (nn.GELU, nn.SiLU, nn.Sigmoid, nn.ReLU))
    return False

def is_transpose(node: Node) -> bool:
    if node.op == 'call_method':
        return node.target in {'transpose', 'transpose_'}
    elif node.op == 'call_function':
        return node.target is torch.transpose
    return False

def is_permute(node: Node) -> bool:
    if node.op == 'call_method':
        return node.target in {'permute'}
    elif node.op == 'call_function':
        return node.target is torch.permute
    return False

def is_getitem(node: Node) -> bool:
    return node.op == 'call_function' and node.target is operator.getitem