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
import glob
import importlib
import json
import os
from functools import partial
from typing import List, Optional, Union

import torch
from torch.fx import GraphModule

from .core import Config, ParallelExecutionCtx
from .passes import build_parallel_pass_pipeline
from .utils import (
    MetaAwareMethodsPatcher,
    convert_bin_to_safetensors,
    download_files_from_hf,
    initialize_parameter_meta,
    move_model_to_device,
)


def parallelize_backend(
    graph_module: GraphModule, example_inputs: List[torch.Tensor], ctx: ParallelExecutionCtx, config: Config
) -> GraphModule:
    ctx.example_inputs = example_inputs
    pass_pipeline = build_parallel_pass_pipeline()
    graph_module = pass_pipeline(graph_module=graph_module, ctx=ctx, config=config)
    ctx.compile_times += 1
    ctx.last_optimized_graph_module = graph_module
    return graph_module


def parallelize_model(
    model: Union[torch.nn.Module, str],
    parallel_ctx: ParallelExecutionCtx,
    *model_args,
    revision: str = "main",
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    API for automatic model parallelism through Pytorch FX.

    Args:
        model (Union[torch.nn.Module, str]):
            Model to parallelize, could either be a module or a model id in huggingface space.
        parallel_ctx (ParallelExecutionCtx):
            Parallel execution context containing process groups the current process belongs to.
        model_args (additional postional arguments, optional):
            Additional postional arguments for intializing the model if a model id is passed.
        revision (`str`, defaults to `main`):
            Model revision for weights downloading if a model id is passed.
        cache_dir (`Optional[str]`, defaults to `None`):
            Cache directory to store downloaded weights. Defaults to None.
        local_files_only (`bool`, defaults to `False`):
            Whether to use local files only, will avoid downloading from remote if set to `True`.
        kwargs (additional keyword arguments, optional):
            Addtional keyword arguments for overriding fields in parallel config, model config and `Model.__init__`.
    """
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

    parallel_config = Config()
    for k, v in kwargs.items():
        if k in parallel_config.__dict__:
            setattr(parallel_config, k, v)
    kwargs = {k: v for k, v in kwargs.items() if k not in parallel_config.__dict__}

    if isinstance(model, str):
        is_local = os.path.isdir(model)
        use_safetensors = False
        allow_patterns = ["*.safetensors", "*.bin"]
        if not is_local:
            hf_folder = download_files_from_hf(
                model_name_or_path=model,
                cache_dir=cache_dir,
                allow_patterns=allow_patterns,
                revision=revision,
                local_files_only=local_files_only,
            )
        else:
            hf_folder = model
        for pattern in allow_patterns:
            if len(glob.glob(os.path.join(hf_folder, pattern))) > 0:
                use_safetensors = pattern == "*.safetensors"
                break
        # should be able to load config using only local files
        model_config, kwargs = AutoConfig.from_pretrained(
            hf_folder, revision=revision, local_files_only=True, return_unused_kwargs=True, **kwargs
        )
        config_path = os.path.join(hf_folder, CONFIG_NAME)
        if not os.path.isfile(config_path):
            raise EnvironmentError(f"Can't find config file {config_path} in {hf_folder}")

        with open(config_path) as f:
            config_dict = json.load(f)
        model_arch = config_dict["architectures"]
        model_cls = getattr(importlib.import_module("transformers"), model_arch[0])

        index_path = os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME)
        if os.path.isfile(index_path):
            with open(index_path) as f:
                index_dict = json.load(f)
            parallel_ctx.weight_map = index_dict["weight_map"]
        weight_files = glob.glob(os.path.join(hf_folder, "*.safetensors" if use_safetensors else "*.bin"))
        if not use_safetensors:
            weight_map = parallel_ctx.weight_map if parallel_ctx.weight_map else {}
            convert_bin_to_safetensors(model, cache_dir, weight_files, weight_map)
            parallel_ctx.weight_map = weight_map

        # try directly construct weight_map from weight files, should have safetensors file on disk in any case
        if not parallel_ctx.weight_map:
            weight_map, weight_files = {}, glob.glob(os.path.join(hf_folder, "*.safetensors"))
            for weight_file in weight_files:
                with safe_open(filename=weight_file, framework="pt") as f:
                    for key in f.keys():
                        weight_map[key] = weight_file
            parallel_ctx.weight_map = weight_map

        with MetaAwareMethodsPatcher():
            model = model_cls(model_config, *model_args, **kwargs)

    move_model_to_device(model, device=parallel_ctx.current_device)
    initialize_parameter_meta(model)
    backend = partial(parallelize_backend, ctx=parallel_ctx, config=parallel_config)
    model = torch.compile(model, fullgraph=True, backend=backend)
    return model
