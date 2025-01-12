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

"""ExecuTorch model check and export functions."""

import logging
import os
from pathlib import Path
from typing import Union

from transformers.utils import is_torch_available

from optimum.utils.import_utils import is_transformers_version

from .recipe_registry import discover_recipes, recipe_registry


if is_torch_available():
    from transformers.modeling_utils import PreTrainedModel

if is_transformers_version(">=", "4.46"):
    from transformers.integrations.executorch import (
        TorchExportableModuleWithStaticCache,
    )

logger = logging.getLogger(__name__)


def export_to_executorch(
    model: Union["PreTrainedModel", "TorchExportableModuleWithStaticCache"],
    task: str,
    recipe: str,
    output_dir: Union[str, Path],
    **kwargs,
):
    """
    Export a pre-trained PyTorch model to the ExecuTorch format using a specified recipe.

    This function facilitates the transformation of a PyTorch model into an optimized ExecuTorch program.

    Args:
        model (`Union["PreTrainedModel", "TorchExportableModuleWithStaticCache"]`):
            A PyTorch model to be exported. This can be a standard HuggingFace `PreTrainedModel` or a wrapped
            module like `TorchExportableModuleWithStaticCache` for text generation task.
        task (`str`):
            The specific task the exported model will perform, e.g., "text-generation".
        recipe (`str`):
            The recipe to guide the export process, e.g., "xnnpack". Recipes define the optimization and lowering steps.
            Will raise an exception if the specified recipe is not registered in the recipe registry.
        output_dir (`Union[str, Path]`):
            Path to the directory where the resulting ExecuTorch model will be saved.
        **kwargs:
            Additional configuration options passed to the recipe.

    Returns:
        `ExecuTorchProgram`:
            The lowered ExecuTorch program object.

    Notes:
        - The function uses a dynamic recipe discovery mechanism to identify and import the specified recipe.
        - The exported model is stored in the specified output directory with the fixed filename `model.pte`.
        - The resulting ExecuTorch program is serialized and saved to the output directory.
    """

    # Dynamically discover and import registered recipes
    discover_recipes()

    # Export and lower the model to ExecuTorch with the recipe
    try:
        recipe_func = recipe_registry.get(recipe)
    except KeyError as e:
        raise RuntimeError(f"The recipe '{recipe}' isn't registered. Detailed error: {e}")

    executorch_prog = recipe_func(model, task, **kwargs)

    full_path = os.path.join(f"{output_dir}", "model.pte")
    with open(full_path, "wb") as f:
        executorch_prog.write_to_file(f)
        logging.info(f"Saved exported program to {full_path}")

    return executorch_prog
