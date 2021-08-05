import operator
from collections import defaultdict

import torch


def _keep_good_candidates(candidates):
    to_keep = set()
    for input_, linears in candidates.items():
        out_features_are_equal = len(set([linear.out_features for _, linear in linears])) == 1
        if len(linears) <= 1 or not out_features_are_equal:
            continue
        to_keep.add(input_)
    return {k: v for (k, v) in candidates.items() if k in to_keep}


def _get_bias(linear):
    if hasattr(linear, "bias"):
        return linear.bias
    return torch.zeros(shape=(linear.out_features), dtype=linear.weight.dtype).to(linear.weight.device)


def _merge_linears(gm, input_node, linear_nodes, linears):
    in_features = linears[0].in_features
    out_features = linears[0].out_features
    total_out_features = len(linears) * out_features
    use_bias = any(hasattr(linear, "bias") for linear in linears)
    merged_linear = torch.nn.Linear(in_features, total_out_features, bias=use_bias)

    with torch.no_grad():
        new_weight = torch.cat([linear.weight for linear in linears], dim=0)
        merged_linear.weight = torch.nn.Parameter(new_weight)
        if use_bias:
            new_bias = torch.cat([_get_bias(linear) for linear in linears], dim=0)
            merged_linear.bias = torch.nn.Parameter(new_bias)

    merged_linear_name = f"{input_node.name}_merged_linear"
    gm.add_module(merged_linear_name, merged_linear)

    graph = gm.graph
    with graph.inserting_after(input_node):
        merged_linear_node = graph.call_module(merged_linear_name, args=(input_node,))

    for idx, node in enumerate(linear_nodes):
        node.op = "call_function"
        node.target = operator.getitem
        slice_to_get = slice(idx * out_features, (idx + 1) * out_features)
        node.args = (merged_linear_node, (Ellipsis, slice_to_get))


def merge_linears(gm):
    """
    Transformation that merges linear layers that take the same input and have the same number of output features.
    """
    candidates = defaultdict(list)
    named_modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            mod = named_modules[node.target]
            if isinstance(mod, torch.nn.Linear):
                input_node = node.args[0]
                candidates[input_node].append((node, mod))

    # Only keep the candidates with more than one linear and the ones with the same number of
    # output features.
    candidates = _keep_good_candidates(candidates)

    for input_node, t in candidates.items():
        linear_nodes, linears = list(zip(*t))
        _merge_linears(gm, input_node, linear_nodes, linears)

    gm.graph.lint()
    gm.recompile()
    return gm


def apply_normalization_factor_to_query(gm):
    """
    Transformation that applies the normalization factor directly to the query weights saving the computation at
    runtime.
    """
    # TODO: add safety checks to make sure that the transformation is applied to attention layers.
    named_modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module" and "query" in node.name:
            p = node
            while p and p.target != operator.truediv:
                p = p.next

            if not isinstance(p.args[0], torch.fx.Node) and p.args[0].target != torch.matmul:
                continue
            if not isinstance(p.args[1], torch.fx.Node):
                query = named_modules[node.target]
                query.weight = torch.nn.Parameter(query.weight / p.args[1])
                if hasattr(query, "bias"):
                    query.bias = torch.nn.Parameter(query.bias / p.args[1])
            p.replace_all_uses_with(p.args[0])

    gm.graph.lint()
    gm.recompile()
    return gm


def optimize_attention(gm):
    """
    Transformation that optimizes attention layers by:

        1. Appliying the normalization factor to the query weights instead of computing it at runtime
        2. Merging the query, key and value linear projections as one big linear projection.
        3. Merging the transpose_for_scores by transposing the output of the merged linear projection instead of doing
           it individually for the query, key and value.
    """
    gm = apply_normalization_factor_to_query(gm)

    graph = gm.graph
    candidates = defaultdict(list)
    named_modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            mod = named_modules[node.target]
            if isinstance(mod, torch.nn.Linear):
                input_node = node.args[0]
                candidates[input_node].append((node, mod))

    # Only keep the candidates with more than one linear and the ones with the same number of
    # output features.
    # TODO: add safety checks to make sure that the transformation is applied to attention layers.
    candidates = _keep_good_candidates(candidates)
    for input_node, t in candidates.items():
        linear_nodes, linears = list(zip(*t))
        _merge_linears(gm, input_node, linear_nodes, linears)

        def find_child_nodes_of_target(node, target_name):
            children = []
            for user in node.users:
                if user.target == target_name:
                    children.append(user)
            return children

        parent_node = linear_nodes[0].args[0]  # output of the linear.
        view_node = find_child_nodes_of_target(linear_nodes[0], "view")
        if len(view_node) != 1:
            continue
        view_node = view_node[0]
        view_shape = list(view_node.args[1:])
        insertion_idx = len(view_shape) - 2
        view_shape.insert(insertion_idx, len(linear_nodes))
        permute_node = find_child_nodes_of_target(view_node, "permute")
        if len(permute_node) != 1:
            continue
        permute_node = permute_node[0]
        permutation = list(permute_node.args[1:])
        permutation = [insertion_idx] + [i if i < insertion_idx else i + 1 for i in permutation]
        with graph.inserting_after(parent_node):
            view_args = tuple([parent_node] + view_shape)
            view_node = graph.call_method("view", args=view_args)
        with graph.inserting_after(view_node):
            permute_args = tuple([view_node] + permutation)
            permute_node = graph.call_method("permute", args=permute_args)
        for idx, node in enumerate(linear_nodes):
            v = find_child_nodes_of_target(node, "view")[0]
            p = find_child_nodes_of_target(v, "permute")[0]
            new_args = [permute_node, (idx,)]
            node.args = tuple(new_args)
            p.replace_all_uses_with(v.args[0])
            graph.erase_node(p)
            graph.erase_node(v)

    gm.graph.lint()
    gm.recompile()
    return gm
