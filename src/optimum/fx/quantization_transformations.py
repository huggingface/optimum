import operator
import functools
from typing import Callable

import torch
from torch.fx import GraphModule


def change_truediv_to_mul_when_possible(gm: GraphModule) -> GraphModule:
    """
    Transforms truediv nodes by multiplication of the inverse when the denominator is static.
    For example, that is sometimes the case for the scaling factor in attention layers.
    """
    graph = gm.graph
    for node in graph.nodes:
        if node.op == "call_function" and node.target == operator.truediv:
            x, y = node.args
            if not isinstance(y, torch.fx.Node):
                node.target = operator.mul
                node.args = (x, 1 / y)
    graph.lint()
    gm.recompile()
    return gm


def change_attention_mask_value(gm: GraphModule, initial_value: int = -10000, final_value: int = -20) -> GraphModule:
    """Changes the attention mask initial value (default is -10000) to some smaller value."""
    graph = gm.graph
    for node in graph.nodes:
        if node.target in [torch.mul, operator.mul] and initial_value in node.args:
            new_args = []
            for arg in node.args:
                if arg == initial_value:
                    new_args.append(final_value)
                else:
                    new_args.append(arg)
            node.args = tuple(new_args)
            break
    graph.lint()
    gm.recompile()
    return gm


def broadcast_attention_mask(gm: GraphModule) -> GraphModule:
    """
    Broadcasts attention_mask to match the shape of attention_scores as broadcasting is not
    supported for quantized add operator in PyTorch.
    """
    graph = gm.graph
    attention_mask = None
    # TODO: find a more robust way of finding the attention_mask node.
    for node in graph.nodes:
        if (
            node.target in [torch.mul, torch.ops.quantized.mul, operator.mul]
            and len(node.args) > 1
            and node.args[1] < 0
        ):
            attention_mask = node
            break
    if attention_mask is None:
        print("Could not find attention_mask to broadcast.")
        return
    for node in graph.nodes:
        if node.target == torch.ops.quantized.add and attention_mask in node.args:
            attention_scores, mask, *rest = node.args
            with graph.inserting_before(node):
                broadcasted_mask = graph.call_function(
                    torch.broadcast_to, args=(mask, graph.call_method("size", args=(attention_scores, )))
                )
                new_args = []
                for arg in node.args:
                    if arg is mask:
                        new_args.append(broadcasted_mask)
                    else:
                        new_args.append(arg)
                node.args = tuple(new_args)
    graph.lint()
    gm.recompile()
    return gm


def broadcast_nonorm_bias(gm: GraphModule) -> GraphModule:
    """
    Broadcasts the NoNorm bias to match the shape of the scaled input as broadcasting is not
    supported for quantized add.
    """
    graph = gm.graph
    for node in graph.nodes:
        if node.target == torch.ops.quantized.mul:
            users = list(node.users.keys())
            if len(users) == 1 and users[0].target == torch.ops.quantized.add:
                add_node = users[0]
                with graph.inserting_before(add_node):
                    broadcasted_bias = graph.call_function(
                        torch.broadcast_to,
                        args=(add_node.args[1], graph.call_method("size", args=(add_node.args[0],))),
                    )
                    new_args = add_node.args[:1] + (broadcasted_bias,) + add_node.args[2:]
                    add_node.args = tuple(new_args)

    graph.lint()
    gm.recompile()
    return gm


def compose_transformations(*args: Callable[[GraphModule], GraphModule]) -> GraphModule:
    """Helper function that allows you to compose transformations together."""
    return functools.reduce(lambda f, g: lambda gm: f(g(gm)), args, lambda x: x)
