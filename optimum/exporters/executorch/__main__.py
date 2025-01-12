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

"""Entry point to the optimum.exporters.executorch command line."""

import argparse
import os
import warnings
from pathlib import Path

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers.utils import is_torch_available

from optimum.utils.import_utils import is_transformers_version

from ...commands.export.executorch import parse_args_executorch
from .convert import export_to_executorch
from .task_registry import discover_tasks, task_registry


if is_torch_available():
    pass

from typing import Optional, Union


def main_export(
    model_name_or_path: str,
    task: str,
    recipe: str,
    output_dir: Union[str, Path],
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    trust_remote_code: bool = False,
    pad_token_id: Optional[int] = None,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    token: Optional[Union[bool, str]] = None,
    **kwargs,
):
    """
    Full-suite ExecuTorch export function, exporting **from a model ID on Hugging Face Hub or a local model repository**.

    Args:
        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the model repository to export. Example: `model_name_or_path="meta-llama/Llama-3.2-1B"` or `mode_name_or_path="/path/to/model_folder`.
        task (`str`):
            The task to export the model for, e.g. "text-generation".
        recipe (`str`):
            The recipe to use to do the export, e.g. "xnnpack".
        output_dir (`Union[str, Path]`):
            Path indicating the directory where to store the generated ExecuTorch model.
        cache_dir (`Optional[str]`, defaults to `None`):
            Path indicating where to store cache. The default Hugging Face cache path will be used by default.
        trust_remote_code (`bool`, defaults to `False`):
            Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
            you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
            model repository.
        pad_token_id (`Optional[int]`, defaults to `None`):
            This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        revision (`str`, defaults to `"main"`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        force_download (`bool`, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        local_files_only (`Optional[bool]`, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
            Deprecated. Please use the `token` argument instead.
        token (`Optional[Union[bool,str]]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
        **kwargs:
            Additional configuration options to tasks and recipes.

    Example usage:
    ```python
    >>> from optimum.exporters.executorch import main_export

    >>> main_export("meta-llama/Llama-3.2-1B", "text-generation", "xnnpack", "meta_llama3_2_1b/")
    ```
    """

    if is_transformers_version("<", "4.46"):
        raise ValueError(
            "The minimum Transformers version compatible with ExecuTorch is 4.46.0. Please upgrade to Transformers 4.46.0 or later."
        )

    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
        token = use_auth_token

    # Dynamically discover and import registered tasks
    discover_tasks()

    # Load the model for specific task
    try:
        task_func = task_registry.get(task)
    except KeyError as e:
        raise RuntimeError(f"The task '{task}' isn't registered. Detailed error: {e}")

    model = task_func(model_name_or_path, **kwargs)

    if task == "text-generation":
        from transformers.integrations.executorch import TorchExportableModuleWithStaticCache

        model = TorchExportableModuleWithStaticCache(model)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return export_to_executorch(
        model=model,
        task=task,
        recipe=recipe,
        output_dir=output_dir,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser("Hugging Face Optimum ExecuTorch exporter")

    parse_args_executorch(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    main_export(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        task=args.task,
        recipe=args.recipe,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        pad_token_id=args.pad_token_id,
    )


if __name__ == "__main__":
    main()
