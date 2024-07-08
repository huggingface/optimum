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
from typing import List

import torch
from torch.fx import GraphModule

from .core import Config, ParallelExecutionCtx
from .passes import build_parallel_pass_pipeline


def parallelize_backend(
    graph_module: GraphModule, example_inputs: List[torch.Tensor], ctx: ParallelExecutionCtx, config: Config
) -> GraphModule:
    ctx.example_inputs = example_inputs
    pass_pipeline = build_parallel_pass_pipeline()
    graph_module = pass_pipeline(graph_module=graph_module, ctx=ctx, config=config)
    ctx.compile_times += 1
    return graph_module
