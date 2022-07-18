import operator
from typing import Any, List

import torch
from torch.fx import Interpreter, GraphModule, Node

from transformers.utils.fx_transformations import transformation, compose_transformations


def change_truediv_to_mul_when_possible_(gm: GraphModule, lint_and_recompile: bool = True):
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

    if lint_and_recompile:
        graph.lint()
        gm.recompile()


def change_attention_mask_value_(
    gm: GraphModule, initial_value: int = -10000, final_value: int = -20, lint_and_recompile: bool = True
):
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

    if lint_and_recompile:
        graph.lint()
        gm.recompile()


class BroadcastAddOps(Interpreter):
    """
    An Interpreter that broadcasts all the operands of operator.add to make them match in terms of shapes.
    This is useful when quantizing a model as torch.ops.quantized.add does not support broadcasting yet.
    """

    def insert_broadcasting_op(
        self, input_to_broadcast: Node, broadcast_size: List[int], other_operand_size_node: Node
    ):
        """
        Inserts a torch.broadcast_to op to broadcast input_to_broadcast on dimensions where broadcast_size is not -1 by
        retrieving the size of the other operand node on these dimensions.
        """
        graph = self.module.graph
        symbolic_broadcast_size = []
        node_to_insert_after = None
        for i, dim in enumerate(broadcast_size):
            if dim != -1:
                with graph.inserting_after(other_operand_size_node):
                    get_ith_item_node = graph.call_function(operator.getitem, args=(other_operand_size_node, i))
                    if node_to_insert_after is None:
                        node_to_insert_after = get_ith_item_node
                    symbolic_broadcast_size.append(get_ith_item_node)
            else:
                symbolic_broadcast_size.append(-1)

        with graph.inserting_after(node_to_insert_after):
            broadcast_op_node = graph.call_function(
                torch.broadcast_to, args=(input_to_broadcast, tuple(symbolic_broadcast_size))
            )

        return broadcast_op_node

    def run_node(self, n: Node) -> Any:
        result = super().run_node(n)
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        graph = self.module.graph

        if n.target is operator.add:
            a, b = args
            shape_a = a.size()
            shape_b = b.size()
            rank_a = len(shape_a)
            rank_b = len(shape_b)
            broadcast_size_a = []
            broadcast_size_b = []
            min_rank = min(rank_a, rank_b)
            max_rank = max(rank_a, rank_b)
            for i in range(min_rank - 1, -1, -1):
                dim_a = shape_a[i]
                dim_b = shape_b[i]
                if dim_a == dim_b:
                    broadcast_size_a.insert(0, -1)
                    broadcast_size_b.insert(0, -1)
                elif dim_a == 1:
                    broadcast_size_a.insert(0, dim_b)
                    broadcast_size_b.insert(0, -1)
                elif dim_b == 1:
                    broadcast_size_a.insert(0, -1)
                    broadcast_size_b.insert(0, dim_a)
                else:
                    # We should never encounter that case.
                    continue

            broadcast_size_a = list(shape_b[: max_rank - rank_a]) + broadcast_size_a
            broadcast_size_b = list(shape_a[: max_rank - rank_b]) + broadcast_size_b

            a_needs_broadcasting = any(dim != -1 for dim in broadcast_size_a)
            b_needs_broadcasting = any(dim != -1 for dim in broadcast_size_b)

            if a_needs_broadcasting or b_needs_broadcasting:
                with graph.inserting_before(n):
                    if b_needs_broadcasting:
                        size_a = graph.call_method("size", args=(n.args[0],))
                    if a_needs_broadcasting:
                        size_b = graph.call_method("size", args=(n.args[1],))

                new_node_args = []

                if a_needs_broadcasting:
                    broadcast_op_node = self.insert_broadcasting_op(n.args[0], broadcast_size_a, size_b)
                    new_node_args.append(broadcast_op_node)
                else:
                    new_node_args.append(n.args[0])

                if b_needs_broadcasting:
                    broadcast_op_node = self.insert_broadcasting_op(n.args[1], broadcast_size_b, size_a)
                    new_node_args.append(broadcast_op_node)
                else:
                    new_node_args.append(n.args[1])

                n.args = tuple(new_node_args)

        return result


@transformation
def broadcast_add(gm: GraphModule) -> GraphModule:
    """Broadcasts operands of operator.add to make them match in terms of shapes."""
    broadcast_add_interpreter = BroadcastAddOps(gm)
    broadcast_add_interpreter.run(*(gm.dummy_inputs.values()))
    gm.graph.lint()
    gm.recompile()
    return gm


def remove_dequantize_after_getitem_on_size_(gm: GraphModule, lint_and_recompile: bool = True):
    """
    Removes .dequantize() ops after operator.getitem that were applied to the result of .size().
    This happens when quantizing a model: the tracer will insert dequantize ops, but because .size() returns
    integers, applying dequantize to this will result in a RuntimeError.
    """
    graph = gm.graph
    for node in graph.nodes:
        if node.op == "call_method" and node.target == "dequantize":
            parent = node.args[0]
            if parent.op == "call_function" and parent.target == operator.getitem:
                grandparent = parent.args[0]
                if grandparent.op == "call_method" and grandparent.target == "size":
                    node.replace_all_uses_with(parent)
                    graph.erase_node(node)
    if lint_and_recompile:
        graph.lint()
        gm.recompile()


pre_quantization_transformations = broadcast_add

post_quantization_transformations = compose_transformations(
    change_attention_mask_value_, remove_dequantize_after_getitem_on_size_
)
