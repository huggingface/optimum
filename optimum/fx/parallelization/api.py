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
import importlib
import os
from functools import partial
from typing import Callable, List, Optional, Type

import torch
import torch.nn as nn
from torch.fx import GraphModule
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from .core import Config, ParallelExecutionCtx
from .utils import (
    MetaAwareMethodsPatcher,
    download_model_from_hf,
    initialize_parameter_meta,
    move_model_to_device,
    try_collect_weight_map,
)


def parallelize_backend(
    graph_module: GraphModule, example_inputs: List[torch.Tensor], ctx: ParallelExecutionCtx, config: Config
) -> nn.Module:
    ctx.example_inputs = example_inputs
    pass_pipeline = ctx.backend.init_parallelization_pass_pipeline()
    graph_module = pass_pipeline(graph_module=graph_module, ctx=ctx, config=config)
    finalized_module = ctx.backend.post_process(graph_module, ctx)
    ctx.compile_times += 1
    ctx.last_optimized_module = finalized_module
    return finalized_module


def parallelize_model(
    parallel_ctx: ParallelExecutionCtx,
    *model_args,
    model_id_or_path: Optional[str] = None,
    model_cls: Optional[Type[PreTrainedModel]] = None,
    model_config: Optional[PretrainedConfig] = None,
    **kwargs,
) -> Callable:
    """
    API for automatic model parallelism through Pytorch FX.

    Args:
        parallel_ctx (`ParallelExecutionCtx`):
            Parallel execution context containing process groups the current process belongs to.
        *model_args (`Any`):
            Additional postional arguments for intializing the model if a model id is passed.
        model_id_or_path (`str`):
            Model to parallelize, a model id on the Huggingface Hub or path to a local directory containing config and weights
            of the model.
        model_cls (`Optional[Type[PreTrainedModel]]`, defaults to `None`):
            Model class in transformers library, i.e, `LlamaForCausalLM`.
        model_config (`Optional[PretrainedConfig]`, defaults to `None`):
            Model config to intialize the model.
        revision (`str`, defaults to `main`):
            Model revision for weights downloading if a model id is passed.
        cache_dir (`Optional[str]`, defaults to `None`):
            Cache directory to store downloaded weights. Defaults to None.
        local_files_only (`bool`, defaults to `False`):
            Whether to use local files only, will avoid downloading from remote if set to `True`.
        skip_load_weights (`bool`, defaults to `False`):
            Whether to skip loading weights from disk to model.
        **kwargs (`Dict[str, Any]`):
            Addtional keyword arguments for overriding fields in parallel config, model config and `Model.__init__`.
    """
    revision = kwargs.pop("revision", "main")
    cache_dir = kwargs.pop("cache_dir", None)
    local_files_only = kwargs.pop("local_files_only", False)
    skip_load_weights = kwargs.pop("skip_load_weights", False)

    parallel_config = Config()
    for k, v in dict(kwargs).items():
        if k in parallel_config.__dict__:
            setattr(parallel_config, k, v)
            kwargs.pop(k)

    if model_id_or_path is not None and (model_cls is not None or model_config is not None):
        raise ValueError(
            "Can not accept passing in all of `model_id_or_path`, `model_cls` and `model_config`. Only specify "
            "`model_id_or_path` or `model_cls` and `model_config` because there might be conflicts otherwise"
        )

    # Init model instance
    if model_id_or_path is not None:
        is_local = os.path.isdir(model_id_or_path)
        if not is_local:
            hf_folder = download_model_from_hf(
                model_name_or_path=model_id_or_path,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
                skip_download_weights=skip_load_weights,
            )
        else:
            hf_folder = model_id_or_path

        # should be able to load config using only local files
        model_config, kwargs = AutoConfig.from_pretrained(
            hf_folder, revision=revision, local_files_only=True, return_unused_kwargs=True, **kwargs
        )

        # try getting model class info from config
        model_arch = model_config.architectures
        model_cls = getattr(importlib.import_module("transformers"), model_arch[0])

        if not skip_load_weights:
            parallel_ctx.weight_map = try_collect_weight_map(model_id_or_path, cache_dir, hf_folder)
    elif model_cls is None or model_config is None:
        raise ValueError("must provide `model_cls` and `model_config` in the case of not providing `model_id_or_path`")

    torch_dtype, dtype_orig = kwargs.pop("torch_dtype", None), None
    if torch_dtype is not None:
        dtype_orig = model_cls._set_default_torch_dtype(torch_dtype)

    with MetaAwareMethodsPatcher():
        model = model_cls(model_config, *model_args, **kwargs)
        # TODO: remove this once support training-time trace
        model.eval()

    if dtype_orig is not None:
        torch.set_default_dtype(dtype_orig)

    move_model_to_device(model, device=parallel_ctx.current_device)
    initialize_parameter_meta(model)
    backend = partial(parallelize_backend, ctx=parallel_ctx, config=parallel_config)
    model = torch.compile(model, fullgraph=True, backend=backend)
    return model
