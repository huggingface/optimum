# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ...utils import logging
from .. import TasksManager
from .base import GgmlConfig


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def export(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, ggml_config: GgmlConfig, output: Path, dtype: str = "fp32"
):
    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    logger.info("the start!")
    output_path = Path(output, "ggml_model.bin")
    fout = output_path.open("wb")

    if dtype == "fp16":
        ftype = 1
    else:
        ftype = 0

    # write ggml in hexadecimal
    hex_string = "ggml".encode("utf-8").hex()
    fout.write(struct.pack("i", int(hex_string, 16)))

    for header_data in ggml_config.header_data:
        fout.write(struct.pack("i", header_data))

    fout.write(struct.pack("i", ftype))

    for i in range(ggml_config._normalized_config.vocab_size):
        text = tokenizer.decode([i]).encode("utf-8")
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    state_dict = model.state_dict()
    state_dict = ggml_config.patch_state_dict(state_dict)

    name_map = ggml_config.get_name_map(state_dict.keys())

    for old_name, new_name in name_map.items():
        logger.info(f"{old_name} -> {new_name}")
        data = state_dict[old_name].squeeze().numpy()
        data = data.astype(np.float32)

        n_dims = len(data.shape)

        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and n_dims > 1:
            logger.info("   Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1

        # header
        new_name_encoded = new_name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(new_name_encoded), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(new_name_encoded)

        # data
        data.tofile(fout)

    fout.close()

    logger.info("The end!")


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str = "auto",
    fp16: Optional[bool] = False,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    do_validation: bool = False,
):
    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    original_task = task
    # Infer the task
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_name_or_path)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder)

    # TODO: GENERALIZE THIS!
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if framework != "pt":
        raise NotImplementedError("ggml export only supports PyTorch checkpoints.")

    torch_dtype = None if fp16 is False else torch.float16
    model = TasksManager.get_model_from_task(
        task,
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
        framework=framework,
        torch_dtype=torch_dtype,
    )

    ggml_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter="ggml", task=task)
    ggml_config = ggml_config_constructor(model.config)

    if original_task == "auto":
        logger.info(f"Automatic task detection to {task}.")

    export(model, tokenizer, ggml_config, output, dtype="fp16" if fp16 else "fp32")

    model.config.save_pretrained(output)

    if do_validation is True:
        raise NotImplementedError("Validation not implemented for ggml")
