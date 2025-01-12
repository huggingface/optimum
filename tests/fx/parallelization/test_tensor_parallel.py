# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from typing import Any, Dict, Union

import torch
import torch.distributed as dist
from dist_utils import NUM_AVAILABLE_DEVICES, SEED, dist_init, gather_at_main_process, spawn, tearDown
from packaging import version
from parameterized import parameterized
from transformers import (
    PretrainedConfig,
    set_seed,
)

from optimum.fx.parallelization import ParallelExecutionCtx, parallelize_model
from optimum.fx.parallelization.parallel_layers import ColumnParallelLinear, VocabParallelEmbedding
from optimum.fx.parallelization.utils import stable_topological_sort


DUMMY_MODEL_KWARGS = {
    "num_hidden_layers": 2,
    "use_cache": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "tie_word_embeddings": True,
    "return_dict": True,
}

DUMMY_MODELS_TO_TEST = (
    (
        "saibo/llama-1B",
        DUMMY_MODEL_KWARGS,
    ),
    (
        "PhoenixJie/dummy-mistral",
        DUMMY_MODEL_KWARGS,
    ),
)


def is_gpu_available():
    return torch.cuda.is_available()


def is_torch_compile_available():
    return version.parse(torch.__version__) >= version.parse("2.3.0")


def prepare_dummy_inputs(
    model_config: PretrainedConfig,
    batch_size: int = 1,
    seq_len: int = 10,
    device: Union[str, torch.device] = "cuda",
):
    input_ids = torch.randint(low=1, high=model_config.vocab_size, size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def run_test_all_rank_results_match(rank: int, world_size: int, model_id: str, model_kwargs: Dict[str, Any]):
    # initialize default group
    dist_init(rank, world_size)
    tp_group = dist.new_group()

    # prepare config and context
    device = torch.device(type="cuda", index=torch.cuda.current_device())
    ctx = ParallelExecutionCtx(tp_group=tp_group, current_device=device)

    model = parallelize_model(model_id, ctx, skip_load_weights=True, **model_kwargs)
    inputs = prepare_dummy_inputs(model.config)
    loss = model(**inputs).loss
    tensors = gather_at_main_process(tensor=loss, group=tp_group, rank=rank, world_size=world_size)

    # check results at main worker process
    if rank == 0:
        assert len(tensors) == world_size
        for i in range(1, world_size):
            torch.testing.assert_close(tensors[i - 1].cpu(), tensors[i].cpu(), rtol=1e-4, atol=1e-4)

    dist.barrier(tp_group)
    tearDown(tp_group)


def run_test_parameters_persist_bewteen_recompile(
    rank: int, world_size: int, model_id: str, model_kwargs: Dict[str, Any]
):
    # initialize default group
    dist_init(rank, world_size)
    tp_group = dist.new_group()

    # prepare config and context
    device = torch.device(type="cuda", index=torch.cuda.current_device())
    ctx = ParallelExecutionCtx(tp_group=tp_group, current_device=device)

    model = parallelize_model(model_id, ctx, skip_load_weights=True, **model_kwargs)
    inputs = prepare_dummy_inputs(model.config)

    # different shape to trigger recompile
    another_inputs = prepare_dummy_inputs(model.config, seq_len=11)
    yet_another_inputs = prepare_dummy_inputs(model.config, batch_size=2, seq_len=12)

    model(**inputs)
    parameter_ids = {id(param) for _, param in ctx.last_optimized_graph_module.named_parameters()}

    model(**another_inputs)
    # check second compilation has been triggered
    assert ctx.compile_times == 2
    parameter_ids_after_recompile = {id(param) for _, param in ctx.last_optimized_graph_module.named_parameters()}
    assert parameter_ids == parameter_ids_after_recompile

    model(**yet_another_inputs)
    assert ctx.compile_times == 3
    parameter_ids_after_recompile = {id(param) for _, param in ctx.last_optimized_graph_module.named_parameters()}
    assert parameter_ids == parameter_ids_after_recompile
    dist.barrier(tp_group)
    tearDown(tp_group)


def run_test_parallel_results_matches_non_parallel(
    rank: int, world_size: int, model_id: str, model_kwargs: Dict[str, Any]
):
    # initialize default group
    dist_init(rank, world_size)
    tp_group = dist.new_group(ranks=[rank])

    # prepare config and context
    device = torch.device(type="cuda", index=torch.cuda.current_device())
    ctx = ParallelExecutionCtx(tp_group=tp_group, current_device=device)

    model = parallelize_model(model_id, ctx, skip_load_weights=True, **model_kwargs)
    inputs = prepare_dummy_inputs(model.config)

    set_seed(SEED)
    loss = model(**inputs).loss

    torch._dynamo.reset()
    del model

    tp_group = dist.new_group()
    set_seed(SEED)
    ctx = ParallelExecutionCtx(tp_group=tp_group, current_device=device)
    model = parallelize_model(model_id, ctx, skip_load_weights=True, **model_kwargs)
    parallel_loss = model(**inputs).loss

    torch.testing.assert_close(loss.cpu(), parallel_loss.cpu(), rtol=1e-4, atol=1e-4)

    dist.barrier(tp_group)
    tearDown()


def run_test_tie_word_embeddings(rank: int, world_size: int, model_id: str, model_kwargs: Dict[str, Any]):
    dist_init(rank, world_size)
    tp_group = dist.new_group()

    # prepare config and context
    device = torch.device(type="cuda", index=torch.cuda.current_device())
    ctx = ParallelExecutionCtx(tp_group=tp_group, current_device=device)
    model = parallelize_model(model_id, ctx, skip_load_weights=True, **model_kwargs)

    inputs = prepare_dummy_inputs(model.config)
    model(**inputs)

    embedding_weight, lm_head_weight = None, None
    graph_module = ctx.last_optimized_graph_module
    stable_topological_sort(graph_module.graph)
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            mod = graph_module.get_submodule(node.target)
            if isinstance(mod, VocabParallelEmbedding):
                embedding_weight = mod.weight
                break
    for node in reversed(graph_module.graph.nodes):
        if node.op == "call_module":
            mod = graph_module.get_submodule(node.target)
            if isinstance(mod, ColumnParallelLinear):
                lm_head_weight = mod.weight
                break
    assert (
        id(embedding_weight) == id(lm_head_weight)
        and hasattr(embedding_weight, "meta")
        and embedding_weight.meta.is_tied
    )
    dist.barrier(tp_group)
    tearDown()


@parameterized.expand(DUMMY_MODELS_TO_TEST)
@unittest.skipIf(
    not is_gpu_available() or not is_torch_compile_available(), "requires gpu and torch version >= 2.3.0 to run"
)
def test_all_rank_results_match(
    model_id,
    model_kwargs,
):
    for world_size in [1, 2, 4, 8]:
        if world_size <= NUM_AVAILABLE_DEVICES:
            spawn(world_size, run_test_all_rank_results_match, model_id, model_kwargs, deterministic=True)


@parameterized.expand(DUMMY_MODELS_TO_TEST)
@unittest.skipIf(
    not is_gpu_available() or not is_torch_compile_available(), "requires gpu and torch version >= 2.3.0 to run"
)
def test_parameters_persist_bewteen_recompile(
    model_id,
    model_kwargs,
):
    for world_size in [1, 2]:
        if world_size <= NUM_AVAILABLE_DEVICES:
            spawn(
                world_size, run_test_parameters_persist_bewteen_recompile, model_id, model_kwargs, deterministic=False
            )


@parameterized.expand(DUMMY_MODELS_TO_TEST)
@unittest.skipIf(
    not is_gpu_available() or not is_torch_compile_available() or NUM_AVAILABLE_DEVICES < 2,
    "requires more than one gpu and torch version >= 2.3.0 to run",
)
def test_parallel_results_matches_non_parallel(
    model_id,
    model_kwargs,
):
    # world_size == 2 is enough
    spawn(2, run_test_parallel_results_matches_non_parallel, model_id, model_kwargs, deterministic=True)


@parameterized.expand(DUMMY_MODELS_TO_TEST)
@unittest.skipIf(
    not is_gpu_available() or not is_torch_compile_available(),
    "requires gpu and torch version >= 2.3.0 to run",
)
def test_tie_word_embeddings(
    model_id,
    model_kwargs,
):
    for world_size in [1, 2]:
        if world_size <= NUM_AVAILABLE_DEVICES:
            spawn(world_size, run_test_tie_word_embeddings, model_id, model_kwargs, deterministic=False)
