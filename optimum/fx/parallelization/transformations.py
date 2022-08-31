import torch.distributed
from torch import nn
from torch.fx import GraphModule

from ..optimization.transformations import Transformation, add_docstring
from .layers import TensorParallelColumnLinear, TensorParallelRowLinear, TensorParallelEmbedding


def match_target_suffix(node_target: str, suffix: str):
    return node_target[-len(suffix) :] == suffix


@add_docstring()
class ApplyTensorParallelismModel(Transformation):
    """
    Transformation that converts a typical single process model into a tensor parallel model
    """

    def __init__(
        self,
        process_group: torch.distributed.ProcessGroup,
        mlp_h_to_4h_target_suffix: str,
        mlp_4h_to_h_target_suffix: str,
        attention_query_key_values_target_suffix: str,
        attention_dense_target_suffix: str,
        lm_head_target_suffix: str,
        word_embeddings_target_suffix: str,
    ):
        super().__init__()

        # # This transformation requires `torch.distributed` to be already initialized
        assert torch.distributed.is_initialized()
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        # We need to define the pattern we match
        self.mlp_h_to_4h_target_suffix = mlp_h_to_4h_target_suffix
        self.mlp_4h_to_h_target_suffix = mlp_4h_to_h_target_suffix
        self.attention_query_key_values_target_suffix = attention_query_key_values_target_suffix
        self.attention_dense_target_suffix = attention_dense_target_suffix
        self.word_embeddings_target_suffix = word_embeddings_target_suffix
        self.lm_head_target_suffix = lm_head_target_suffix

        # TODO @thomasw21: For now we don't apply parallel softmax to lm_head, maybe we should, or gather right before.
        self.column_parallels = [self.mlp_h_to_4h_target_suffix, self.attention_query_key_values_target_suffix, self.lm_head_target_suffix]
        self.row_parallels = [self.mlp_4h_to_h_target_suffix, self.attention_dense_target_suffix]

    def get_new_word_embeddings(self, module: nn.Module) -> nn.Module:
        assert isinstance(module, nn.Embedding)

        vocab_size, hidden_dim = module.weight.shape
        new_module = TensorParallelEmbedding(
            vocab_size,
            hidden_dim,
            process_group = self.process_group,
            padding_idx = module.padding_idx,
            max_norm = module.max_norm,
            norm_type = module.norm_type,
            scale_grad_by_freq = module.scale_grad_by_freq,
            sparse = module.sparse,
            _weight = None,
            device = module.device,
            dtype = module.dtype,
        )

        new_module.weight = nn.Parameter(nn.module.weight.detach()[new_module.min_id: new_module.max_id, :].clone())

        return new_module

    def get_new_column_linear(self, module: nn.Module) -> nn.Module:
        assert isinstance(module, nn.Linear), "We should only be replacing `nn.Linear` layers"

        use_bias = module.bias is not None
        # Change weights
        new_module = TensorParallelColumnLinear(
            module.in_features,
            module.out_features,
            process_group=self.process_group,
            bias=use_bias,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )

        new_module.weight = torch.nn.Parameter(
            module.weight.detach()[self.tp_rank * new_module.block_size: (self.tp_rank + 1) * new_module.block_size, :].clone()
        )
        if use_bias:
            new_module.bias = torch.nn.Parameter(
                module.bias.detach()[self.tp_rank * new_module.block_size: (self.tp_rank + 1) * new_module.block_size].clone()
            )

        return new_module

    def get_new_row_linear(self, module: nn.Module) -> nn.Module:
        assert isinstance(module, nn.Linear), "We should only be replacing `nn.Linear` layers"

        use_bias = module.bias is not None
        new_module = TensorParallelRowLinear(
            module.in_features,
            module.out_features,
            process_group=self.process_group,
            bias=use_bias,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )

        new_module.weight = torch.nn.Parameter(
            module.weight.detach()[:, self.tp_rank * new_module.block_size: (self.tp_rank + 1) * new_module.block_size].clone()
        )
        if use_bias:
            if self.tp_rank == 0:
                new_module.bias = torch.nn.Parameter(module.bias.detach().clone())
            else:
                new_module.bias.data.zero_()

        return new_module

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        # We find pattens
        layers_to_parallelise_per_target_suffix = {
            self.mlp_h_to_4h_target_suffix: [],
            self.mlp_4h_to_h_target_suffix: [],
            self.attention_query_key_values_target_suffix: [],
            self.attention_dense_target_suffix: [],
            self.lm_head_target_suffix: [],
            self.word_embeddings_target_suffix: [],
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
                if suffix == self.word_embeddings_target_suffix:
                    get_new_module = self.get_new_word_embeddings
                elif suffix in self.column_parallels:
                    get_new_module = self.get_new_column_linear
                elif suffix in self.row_parallels:
                    get_new_module = self.get_new_row_linear
                else:
                    raise ValueError(f"I don't know what to do with {suffix}")

                for candidate in candidates:
                    dirname_and_basename = candidate.target.rsplit(".", maxsplit=1)
                    if len(dirname_and_basename) == 2:
                        fully_qualified_parent_name, module_name = dirname_and_basename
                        parent_module = graph_module.get_submodule(fully_qualified_parent_name)
                    else:
                        assert len(dirname_and_basename) == 1
                        module_name = dirname_and_basename[0]
                        parent_module = graph_module

                    module = getattr(parent_module, module_name)

                    new_module = get_new_module(module)

                    setattr(parent_module, module_name, new_module)
                    del module


        # We find `num_heads` nodes used as input. key -> `num_heads` node, value -> node using it as input
        nodes_using_num_head_nodes_as_input = {}
        for node in graph_module.graph.nodes:
            for input_node in node.all_input_nodes:
                if input_node.op != "get_attr":
                    continue

                path = input_node.target.rsplit(".")
                if path[-1] in ["num_heads", "n_head"]:
                    # TODO @thomasw21: some module require the original value of `num_heads`, typically alibi constructor
                    if "self_attention" in path:
                        if input_node in nodes_using_num_head_nodes_as_input:
                            nodes_using_num_head_nodes_as_input[input_node].append(node)
                        else:
                            nodes_using_num_head_nodes_as_input[input_node] = [node]

        # add a `/ tp_world_size` after each num_heads:
        for num_head_node, nodes in nodes_using_num_head_nodes_as_input.items():
            with graph_module.graph.inserting_after(num_head_node):
                new_head_node = graph_module.graph.create_node("call_method", "__floordiv__", (num_head_node, self.tp_world_size))

            for node in nodes:
                node.replace_input_with(num_head_node, new_head_node)

        return graph_module


@add_docstring()
class ApplyTensorParallelismAlibi(Transformation):
    """
    Alibi is not independent of tensor parallelism, we device a hack to find it.
    TODO @thomasw21: make it so that it's not a hack.
    """

    def __init__(
        self,
        process_group: torch.distributed.ProcessGroup,
    ):
        super().__init__()
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        # we find `torch.baddbmm` nodes
        baddbmm_nodes = []
        for node in graph_module.graph.nodes:
            if node.op != "call_method":
                continue

            # HACK @thomasw21: alibi is the matrix use right before baddbmm
            if node.target == "baddbmm":
                baddbmm_nodes.append(node)

        # We make sure that all `torch.baddbmm` nodes have the exact same parent, and that parent is alibi
        parents = [baddbmm_node.args[0] for baddbmm_node in baddbmm_nodes]
        assert all(parents[0] == elt for elt in parents)
        cast_alibi = parents[0]  # this is a casting to fp32
        reshape_alibi = cast_alibi.args[0]  # this is a reshape `.reshape(batch_size * num_heads, 1, seq_length)`
        unreshaped_alibi = reshape_alibi.args[0]

        # we need to add a node where we get a specific shard
        # TODO @thomasw21: Maybe I need to trace that function
        def slice_alibi(alibi):
            batch_size, num_heads, seq_length = alibi.shape
            assert num_heads % self.tp_world_size == 0
            block_size = num_heads // self.tp_world_size
            return alibi[:, self.tp_rank * block_size : (self.tp_rank + 1) * block_size].contiguous().view(
                batch_size * block_size, 1, seq_length
            )

        with graph_module.graph.inserting_after(unreshaped_alibi):
            new_alibi_node = graph_module.graph.create_node("call_function", slice_alibi, (unreshaped_alibi,))

        # Plug the new alibi
        cast_alibi.replace_input_with(reshape_alibi, new_alibi_node)

        # Remove old node:
        graph_module.graph.erase_node(reshape_alibi)
        del reshape_alibi

        return graph_module

