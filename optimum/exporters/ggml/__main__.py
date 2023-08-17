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
from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import AutoConfig, AutoTokenizer

from optimum.commands.export.ggml import parse_args_ggml
from optimum.exporters.ggml.utils import bytes_to_unicode, infer_task
from optimum.exporters.tasks import TasksManager
from optimum.utils import logging


logger = logging.get_logger()
logger.setLevel(logging.INFO)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel


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
) -> Union["PreTrainedModel", "TFPreTrainedModel", None]:
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

    fname_out = os.path.join(output, f"ggml-model-{model_name_or_path.split('/')[-1]}-{ftype_str[ftype]}.bin")
    fout = open(fname_out, "wb")

    # Hardcoded for Bloom TODO remove as argument in cpp and hardcode there so hparam can be removed
    hparams["multiple_of"] = 1

    vocab_size = hparams["vocab_size"]

    fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex
    fout.write(struct.pack("i", vocab_size))
    for key in ggml_config.STRUCT_HPARAM_KEYS:
        fout.write(struct.pack("i", hparams[key]))
    fout.write(struct.pack("i", ftype))

    if ggml_config.USE_BYTE_DECODER:
        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        encoder = tokenizer.vocab

        fout.write(struct.pack("i", vocab_size))

        for key in sorted(encoder, key=encoder.get):
            text = bytearray([byte_decoder[c] for c in key])
            fout.write(struct.pack("i", len(text)))
            fout.write(text)

    else:
        for i in range(vocab_size):
            text = tokenizer.decode([i]).encode("utf-8")
            fout.write(struct.pack("i", len(text)))
            fout.write(text)

    list_vars = model.state_dict()
    for name in list_vars.keys():
        print("Processing variable: " + name)

        if hasattr(ggml_config, "get_cpp_name"):
            cpp_name = ggml_config.get_cpp_name(name=name)

        if hasattr(ggml_config, "should_skip") and ggml_config.should_skip(name=cpp_name):
            continue

        if hasattr(ggml_config, "reshape_weights"):
            list_vars[name] = ggml_config.reshape_weights(name=cpp_name, weights=list_vars[name], hparams=hparams)

        n_dims = len(list_vars[name].shape)
        data, ftype_cur = ggml_config.convert_dtype(name=cpp_name, data=list_vars[name], ftype=ftype, n_dims=n_dims)

        if data.nbytes % ggml_config.GGML_MEM_ALIGN != 0:
            description = f"Expected data (weights of {name}) to have a multiple of f{ggml_config.GGML_MEM_ALIGN} bytes, but data has {data.nbytes} bytes. Skipping to avoid memory alignment issues."
            print(f"  {description}")
            logger.warning(description)
            continue

        # header
        str = cpp_name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: " + fname_out)
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
