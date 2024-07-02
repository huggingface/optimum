import unittest
import torch
import torch.distributed as dist
from typing import Type
from functools import partial
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    LlamaConfig,
    MistralConfig,
    LlamaForCausalLM,
    MistralForCausalLM,
    set_seed,
)
from parameterized import parameterized
from optimum.fx.parallelization import parallelize_backend, ParallelExecutionCtx, Config
from optimum.fx.parallelization.utils import MetaAwareMethodsPatcher, move_model_to_device, initialize_parameter_mapping
from dist_utils import (
    dist_init,
    tearDown,
    spawn,
    gather_at_main_process,
    NUM_AVAILABLE_DEVICES,
    SEED
)


DUMMY_MODELS_TO_TEST = (
    (LlamaForCausalLM, LlamaConfig(), ),
    (MistralForCausalLM, MistralConfig(), ),
)


def dummify(config: PretrainedConfig):
    config.num_hidden_layers = 2
    config.use_cache = False
    config.output_attentions = False
    config.output_hidden_states = False

def run_test_all_rank_results_match(rank: int, world_size: int, model_cls: Type[PreTrainedModel], model_config: PretrainedConfig):
    dummify(model_config)

    # initialize default group
    dist_init(rank, world_size)
    tp_group = dist.new_group()
    
    # prepare config and context
    device = torch.device(type='cuda', index=torch.cuda.current_device())
    ctx, cfg = ParallelExecutionCtx(tp_group=tp_group, current_device=device), Config()

    inputs = {
        "input_ids": torch.randint(low=1, high=model_config.vocab_size, size=(1, 10), device=device),
        "attention_mask": torch.ones((1, 10), dtype=torch.int64, device=device),
        "position_ids": torch.arange(0, 10, device=device).unsqueeze(0),
    }

    # this will initialize all linears on meta device
    with MetaAwareMethodsPatcher():
        model = model_cls(model_config)
        model.eval()
    # move model to current device, with linears still on meta, and intialize parameter mapping
    move_model_to_device(model, device=device)
    initialize_parameter_mapping(model, ctx=ctx)

    model = torch.compile(model, fullgraph=True, backend=partial(parallelize_backend, ctx=ctx, config=cfg))
    logits = model(**inputs)[0]
    tensors = gather_at_main_process(tensor=logits, group=tp_group, rank=rank, world_size=world_size)

    # check results at main worker process
    if rank == 0:
        assert len(tensors) == world_size
        for i in range(1, world_size):
            torch.testing.assert_close(tensors[i - 1].cpu(), tensors[i].cpu(), rtol=1e-4, atol=1e-4)
    
    dist.barrier(tp_group)
    tearDown(tp_group)

def run_test_parameters_persist_bewteen_recompile(rank: int, world_size: int, model_cls: Type[PreTrainedModel], model_config: PretrainedConfig):
    dummify(model_config)

    # initialize default group
    dist_init(rank, world_size)
    tp_group = dist.new_group()
    
    # prepare config and context
    device = torch.device(type='cuda', index=torch.cuda.current_device())
    ctx, cfg = ParallelExecutionCtx(tp_group=tp_group, current_device=device), Config()

    inputs = {
        "input_ids": torch.randint(low=1, high=model_config.vocab_size, size=(1, 10), device=device),
        "attention_mask": torch.ones((1, 10), dtype=torch.int64, device=device),
        "position_ids": torch.arange(0, 10, device=device).unsqueeze(0),
    }
    
    # different shape to trigger recompile
    another_inputs = {
        "input_ids": torch.randint(low=1, high=model_config.vocab_size, size=(1, 11), device=device),
        "attention_mask": torch.ones((1, 11), dtype=torch.int64, device=device),
        "position_ids": torch.arange(0, 11, device=device).unsqueeze(0),
    }

    # this will initialize all linears on meta device
    with MetaAwareMethodsPatcher():
        model = model_cls(model_config)
        model.eval()
    # move model to current device, with linears still on meta
    move_model_to_device(model, device=device)
    initialize_parameter_mapping(model, ctx=ctx)

    model = torch.compile(model, fullgraph=True, backend=partial(parallelize_backend, ctx=ctx, config=cfg))
    model(**inputs)

    parameter_ids = set([id(param) for _, param in model.named_parameters()])
    model(**another_inputs)

    parameter_ids_after_recompile = set([id(param) for _, param in model.named_parameters()])
    assert parameter_ids == parameter_ids_after_recompile

    dist.barrier(tp_group)
    tearDown(tp_group)

def run_test_parallel_results_matches_non_parallel(rank: int, world_size: int, model_cls: Type[PreTrainedModel], model_config: PretrainedConfig):
    dummify(model_config)

    dist_init(rank, world_size)
    tp_group = dist.new_group(ranks=[rank])
    
    # prepare config and context
    device = torch.device(type='cuda', index=torch.cuda.current_device())
    ctx, cfg = ParallelExecutionCtx(tp_group=tp_group, current_device=device), Config()
    
    inputs = {
        "input_ids": torch.randint(low=1, high=model_config.vocab_size, size=(1, 10), device=device),
        "attention_mask": torch.ones((1, 10), dtype=torch.int64, device=device),
        "position_ids": torch.arange(0, 10, device=device).unsqueeze(0),
    }

    set_seed(SEED)
    # non-parallel local forward
    with MetaAwareMethodsPatcher():
        model = model_cls(model_config)
        model.eval()

    # move model to current device, with linears still on meta
    move_model_to_device(model, device=device)
    initialize_parameter_mapping(model, ctx=ctx)

    model = torch.compile(model, fullgraph=True, backend=partial(parallelize_backend, ctx=ctx, config=cfg))
    logits = model(**inputs)[0]

    del model

    tp_group = dist.new_group()
    set_seed(SEED)
    ctx = ParallelExecutionCtx(tp_group=tp_group, current_device=device)
    with MetaAwareMethodsPatcher():
        model = model_cls(model_config)
        model.eval()

    # move model to current device, with linears still on meta
    move_model_to_device(model, device=device)
    initialize_parameter_mapping(model, ctx=ctx)

    model = torch.compile(model, fullgraph=True, backend=partial(parallelize_backend, ctx=ctx, config=cfg))
    parallel_logits = model(**inputs)[0]

    torch.testing.assert_close(logits.cpu(), parallel_logits.cpu(), rtol=1e-4, atol=1e-4)

    dist.barrier(tp_group)
    tearDown()

@parameterized.expand(DUMMY_MODELS_TO_TEST)
@unittest.skipIf(not torch.cuda.is_available(), "requires gpu to run")
def test_all_rank_results_match(model_cls, config, ):
    for world_size in [1, 2, 4, 8]:
        if world_size <= NUM_AVAILABLE_DEVICES:
            spawn(world_size, run_test_all_rank_results_match, model_cls, config, deterministic=True)

@parameterized.expand(DUMMY_MODELS_TO_TEST)
@unittest.skipIf(not torch.cuda.is_available(), "requires gpu to run")
def test_parameters_persist_bewteen_recompile(model_cls, config, ):
    for world_size in [1, 2, 4, 8]:
        if world_size <= NUM_AVAILABLE_DEVICES:
            spawn(world_size, run_test_parameters_persist_bewteen_recompile, model_cls, config, deterministic=False)

@parameterized.expand(DUMMY_MODELS_TO_TEST)
@unittest.skipIf(not torch.cuda.is_available(), "requires gpu to run")
def test_parallel_results_matches_non_parallel(model_cls, config, ):
    # world_size == 2 is enough
    spawn(2, run_test_parallel_results_matches_non_parallel, model_cls, config, deterministic=True)