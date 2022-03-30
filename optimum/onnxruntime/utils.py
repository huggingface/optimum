#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.utils import logging

import onnx


logger = logging.get_logger(__name__)


class ORTConfigManager:
    """
    A class that contains all the information needed by ONNX Runtime optimization for a given model type.

    Attributes:
        _conf (`Dict[str, tuple]`):
            A dictionary mapping each supported model type to a tuple containing the number of attention heads
            and the hidden size model config attribute names as well as the corresponding ONNX Runtime model type.
    """

    _conf = {
        "bert": ("num_attention_heads", "hidden_size", "bert"),
        "distilbert": ("n_heads", "hidden_size", "bert"),
        "roberta": ("num_attention_heads", "hidden_size", "bert"),
        "camembert": ("num_attention_heads", "hidden_size", "bert"),
        "albert": ("num_attention_heads", "hidden_size", "bert"),
        "bart": ("encoder_attention_heads", "d_model", "bart"),
        "gpt2": ("n_head", "n_embd", "gpt2"),
        "gpt_neo": ("num_heads", "hidden_size", "gpt2"),
    }

    @classmethod
    def get_num_heads_name(cls, model_type: str) -> str:
        num_heads = "num_attention_heads"
        try:
            num_heads = cls._conf[model_type][0]
        except KeyError:
            logger.warning(
                f"{model_type} is not supported yet. Only {list(cls._conf.keys())} are supported. The default value to "
                f"access the number of heads defined in the config is set to `{num_heads}`."
            )
        return num_heads

    @classmethod
    def get_hidden_size_name(cls, model_type: str) -> str:
        hidden_size = "hidden_size"
        try:
            hidden_size = cls._conf[model_type][1]
        except KeyError:
            logger.warning(
                f"{model_type} is not supported yet. Only {list(cls._conf.keys())} are supported. The default value to "
                f"access the hidden size defined in the config is set to `{hidden_size}`."
            )
        return hidden_size

    @classmethod
    def get_model_ort_type(cls, model_type: str) -> str:
        try:
            model_type = cls._conf[model_type][2]
        except KeyError:
            logger.warning(f"{model_type} is not supported yet. Only {list(cls._conf.keys())} are supported.")
        return model_type

    @classmethod
    def check_supported_model_or_raise(cls, model_type: str) -> bool:
        if model_type not in cls._conf:
            raise KeyError(
                f"{model_type} model type is not supported yet. Only {list(cls._conf.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )


def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


def fix_atenops_to_gather(model_path):
    # Fix broken ATenOp nodes back to Gather nodes.
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    nodes = model.graph.node

    for node in nodes:
        if node.op_type in ["ATenOp", "ATen"]:
            logger.info(f"----Start fixing node: {node.name}----")
            op_num = node.name.split("_")[-1]
            new_node = onnx.helper.make_node(
                "Gather",
                name="Gather_" + op_num,
                inputs=[node.input[0], node.input[1]],
                outputs=node.output,
            )

            model.graph.node.remove(node)
            model.graph.node.insert(int(op_num), new_node)

    onnx.checker.check_model(model)
    onnx.save(model, model_path)
