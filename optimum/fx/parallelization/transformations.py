import torch.distributed
from torch import nn
from torch.fx import GraphModule

from ..optimization.transformations import Transformation, add_docstring
from .layers import TensorParallelColumnLinear, TensorParallelRowLinear


def match_target_suffix(node_target: str, suffix: str):
    return node_target[-len(suffix) :] == suffix


@add_docstring()
class ApplyTensorParallelismModel(Transformation):
    """
    Transformation that converts a typical single process model into a tensor parallel model
    """

    def __init__(
        self,
        tp_rank: int,
        tp_world_size: int,
        process_group: torch.distributed.ProcessGroup,
        mlp_h_to_4h_target_suffix: str,
        mlp_4h_to_h_target_suffix: str,
        attention_query_key_values_target_suffix: str,
        attention_dense_target_suffix: str,
    ):
        super().__init__()

        # # This transformation requires `torch.distributed` to be already initialized
        # assert torch.distributed.is_initialized()
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.process_group = process_group

        # We need to define the pattern we match
        self.mlp_h_to_4h_target_suffix = mlp_h_to_4h_target_suffix
        self.mlp_4h_to_h_target_suffix = mlp_4h_to_h_target_suffix
        self.attention_query_key_values_target_suffix = attention_query_key_values_target_suffix
        self.attention_dense_target_suffix = attention_dense_target_suffix
        # self.embedding_layer_target_suffix =
        # self.lm_head_target_suffix =

        self.column_parallels = [self.mlp_h_to_4h_target_suffix, self.attention_query_key_values_target_suffix]
        self.row_parallels = [self.mlp_4h_to_h_target_suffix, self.attention_dense_target_suffix]

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        # We find pattens
        layers_to_parallelise_per_target_suffix = {
            self.mlp_h_to_4h_target_suffix: [],
            self.mlp_4h_to_h_target_suffix: [],
            self.attention_query_key_values_target_suffix: [],
            self.attention_dense_target_suffix: [],
        }
        for node in graph_module.graph.nodes:
            # Filter out all non module nodes.
            if node.op != "call_module":
                continue

            # Append all nodes that match each suffix
            for suffix, buffer in layers_to_parallelise_per_target_suffix.items():
                if match_target_suffix(node.target, suffix):
                    buffer.append(node)
                    break

        # we replace patterns with replacement
        with torch.no_grad():
            for suffix, candidates in layers_to_parallelise_per_target_suffix.items():
                for candidate in candidates:
                    fully_qualified_parent_name, module_name = candidate.target.rsplit(".", maxsplit=1)
                    parent_module = graph_module.get_submodule(fully_qualified_parent_name)

                    module = getattr(parent_module, module_name)
                    assert isinstance(module, nn.Linear), "We should only be replacing `nn.Linear` layers"

                    output_dim, input_dim = module.weight.shape
                    use_bias = module.bias is not None
                    if suffix in self.row_parallels:
                        # Change weights
                        assert input_dim % self.tp_world_size == 0
                        block_size = input_dim // self.tp_world_size
                        new_module = TensorParallelRowLinear(
                            block_size,
                            output_dim,
                            process_group=self.process_group,
                            bias=use_bias,
                            dtype=module.weight.dtype,
                            device=module.weight.device,
                        )

                        new_module.weight = torch.nn.Parameter(
                            module.weight[:, self.tp_rank * block_size : (self.tp_rank + 1) * block_size].clone()
                        )
                        if use_bias:
                            if self.tp_rank == 0:
                                new_module.bias = torch.nn.Parameter(module.bias.clone())
                            else:
                                new_module.bias.data.zero_()
                    else:
                        # Change weights
                        assert suffix in self.column_parallels, f"{suffix} not in {self.column_parallels}"
                        assert output_dim % self.tp_world_size == 0
                        block_size = output_dim // self.tp_world_size
                        new_module = TensorParallelColumnLinear(
                            input_dim,
                            block_size,
                            process_group=self.process_group,
                            bias=use_bias,
                            dtype=module.weight.dtype,
                            device=module.weight.device,
                        )

                        new_module.weight = torch.nn.Parameter(
                            module.weight[self.tp_rank * block_size : (self.tp_rank + 1) * block_size, :].clone()
                        )
                        if use_bias:
                            new_module.bias = torch.nn.Parameter(
                                module.bias[self.tp_rank * block_size : (self.tp_rank + 1) * block_size].clone()
                            )

                    setattr(parent_module, module_name, new_module)
                    del module

        return graph_module


@add_docstring()
class ApplyTensorParallelismAlibi(Transformation):
    """
    Alibi is not independent of tensor parallelism, we device a hack to find it.
    TODO @thomasw21: make it so that it's not a hack.
    """

    def __init__(
        self,
        tp_rank: int,
        tp_world_size: int,
        process_group: torch.distributed.ProcessGroup,
    ):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        # we find `torch.baddbmm` nodes
        baddbmm_nodes = []
        for node in graph_module.graph.nodes:
            if node.op != "call_method":
                continue

            if node.target == "baddbmm":
                baddbmm_nodes.append(node)

        # We make sure that all `torch.baddbmm` nodes have the exact same parent, and that parent is alibi
        parents = [baddbmm_node.args[0] for baddbmm_node in baddbmm_nodes]
        assert all(parents[0] == elt for elt in parents)
        cast_alibi = parents[0]  # this is a casting to fp32
        reshape_alibi = cast_alibi.args[0]  # this is a reshape `.reshape(batch_size * num_heads, 1, seq_length)`
        unreshaped_alibi = reshape_alibi.args[0]

        if self.tp_rank == 0:
            print(cast_alibi.name, cast_alibi.op, cast_alibi.target, cast_alibi.args, cast_alibi.kwargs)
            print(reshape_alibi.name, reshape_alibi.op, reshape_alibi.target, reshape_alibi.args, reshape_alibi.kwargs)
            print(
                unreshaped_alibi.name,
                unreshaped_alibi.op,
                unreshaped_alibi.target,
                unreshaped_alibi.args,
                unreshaped_alibi.kwargs,
            )

        # we need to add a node where we get a specific shard
        # TODO @thomasw21: Maybe I need to trace that function
        def slice_alibi(alibi):
            batch_size, num_heads, seq_length = alibi.shape
            assert num_heads % self.tp_world_size == 0
            block_size = num_heads // self.tp_world_size
            return alibi[:, self.tp_rank * block_size : (self.tp_rank + 1) * block_size].reshape(
                batch_size * block_size, 1, seq_length
            )

        with graph_module.graph.inserting_after(unreshaped_alibi):
            new_alibi_node = graph_module.graph.create_node("call_function", slice_alibi, (unreshaped_alibi,))

        # Plug the new alibi
        cast_alibi.args = (new_alibi_node, *cast_alibi.args[1:])

        # Remove all node:
        del reshape_alibi

        return graph_module
