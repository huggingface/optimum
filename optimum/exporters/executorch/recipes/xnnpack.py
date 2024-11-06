# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from typing import Union

import torch
import torch.export._trace
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from torch.nn.attention import SDPBackend
from transformers import PreTrainedModel, TorchExportableModuleWithStaticCache

from ..recipe_registry import register_recipe


@register_recipe("xnnpack")
def export_to_executorch_with_xnnpack(
    model: Union[PreTrainedModel, TorchExportableModuleWithStaticCache],
    task: str,
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to XNNPACK backend.

    This function also write metadata required by the ExecuTorch runtime to the model.

    Args:
        model (Union[PreTrainedModel, TorchExportableModuleWithStaticCache]):
            The PyTorch model to be exported to ExecuTorch.
        task (str):
            The task name to export the model for (e.g., "text-generation").
        **kwargs:
            Additional keyword arguments for recipe-specific configurations.

    Returns:
        ExecuTorchProgram:
            The exported and optimized program for ExecuTorch.
    """
    metadata = {}
    if task == "text-generation":
        example_input_ids = torch.tensor([[1]], dtype=torch.long)
        example_cache_position = torch.tensor([0], dtype=torch.long)

        def _get_constant_methods(model: PreTrainedModel):
            metadata = {
                "get_dtype": 5 if model.config.torch_dtype == torch.float16 else 6,
                "get_bos_id": model.config.bos_token_id,
                "get_eos_id": model.config.eos_token_id,
                "get_head_dim": model.config.hidden_size / model.config.num_attention_heads,
                "get_max_batch_size": model.generation_config.cache_config.batch_size,
                "get_max_seq_len": model.generation_config.cache_config.max_cache_len,
                "get_n_kv_heads": model.config.num_key_value_heads,
                "get_n_layers": model.config.num_hidden_layers,
                "get_vocab_size": model.config.vocab_size,
                "use_kv_cache": model.generation_config.use_cache,
            }
            return {k: v for k, v in metadata.items() if v is not None}

        metadata = _get_constant_methods(model if isinstance(model, PreTrainedModel) else model.model)
    else:
        # TODO: Prepare model inputs for other tasks
        raise ValueError(f"Unsupported task '{task}'.")

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        exported_program = torch.export._trace._export(
            model,
            args=(example_input_ids,),
            kwargs={"cache_position": example_cache_position},
            pre_dispatch=False,
            strict=True,
        )

        return to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
        ).to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=True,
            ),
        )
