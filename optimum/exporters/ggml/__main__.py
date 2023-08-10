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
"""Entry point to the optimum.exporters.ggml command line."""

import os
import struct
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from optimum.commands.export.ggml import parse_args_ggml
from optimum.exporters.ggml.utils import infer_task
from optimum.exporters.tasks import TasksManager
from optimum.utils import logging


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def _get_submodels_and_ggml_configs(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    task: str,
):
    ggml_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter="ggml", task=task)
    ggml_config = ggml_config_constructor(model.config)

    return ggml_config


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str = "auto",
    fp16: Optional[bool] = False,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    return_source_model: bool = False,
):
    """
    Full-suite ggml export.
    """

    output = Path(output)

    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    # Infer the task
    task = infer_task(model_name_or_path, task)

    # make sure the output directory exists
    os.makedirs(output, exist_ok=True)

    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    #
    # map from ftype to string
    ftype_str = ["f32", "f16"]
    ftype = int(fp16)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    hparams = config.to_dict()

    model = TasksManager.get_model_from_task(
        task,
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        config=config,
        torch_dtype=torch.float16 if ftype == 1 else torch.float32,
    )

    ggml_config = _get_submodels_and_ggml_configs(
        model=model,
        task=task,
    )

    conv_map = ggml_config.CONV_MAP

    fname_out = os.path.join(output, f"ggml-model-{model_name_or_path.split('/')[-1]}-{ftype_str[ftype]}.bin")
    fout = open(fname_out, "wb")

    hparams["multiple_of"] = 1
    fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["n_positions"]))
    fout.write(struct.pack("i", hparams["hidden_size"]))
    fout.write(struct.pack("i", hparams["multiple_of"]))
    fout.write(struct.pack("i", hparams["n_head"]))
    fout.write(struct.pack("i", hparams["n_layer"]))
    fout.write(struct.pack("i", ftype))

    for i in range(hparams["vocab_size"]):
        text = tokenizer.decode([i]).encode("utf-8")
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    list_vars = model.state_dict()
    for name in list_vars.keys():
        src = name
        nn = name
        if name != "lm_head.weight":
            nn = nn.split(".")[1:]
        else:
            nn = nn.split(".")

        if nn[0] == "h":
            nn[0] = "layers"
            mapped = conv_map[".".join(nn[2:-1])]
            name = ".".join(nn[:2] + [mapped] + nn[-1:])
        else:
            mapped = conv_map[".".join(nn[:-1])]
            name = ".".join([mapped] + nn[-1:])

        if "query_key_value" in src:
            q, k, v = list_vars[src].reshape(config.n_head, 3, -1).unbind(1)
            list_vars[src] = torch.cat([q, k, v], dim=0).reshape_as(list_vars[src])

        print(src, " -> ", name)
        data = list_vars[src].squeeze().numpy()
        data = data.astype(np.float32)

        n_dims = len(data.shape)
        print(name, n_dims, data.shape)

        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and n_dims > 1:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1

        # header
        str = name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()

    if return_source_model:
        return model


def main():
    parser = ArgumentParser("Hugging Face Optimum ggml exporter")

    parse_args_ggml(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    main_export(
        model_name_or_path=args.model,
        output=args.output,
        cache_dir=args.cache_dir,
        task=args.task,
    )


if __name__ == "__main__":
    main()
