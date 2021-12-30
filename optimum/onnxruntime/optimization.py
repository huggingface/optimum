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

import logging
import os
from pathlib import Path
from typing import Callable, Optional, Union
from collections import OrderedDict

from transformers import AutoTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

import onnx
from onnx import load_model
from onnxruntime.transformers.optimizer import optimize_model, get_fusion_statistics, FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from optimum.onnxruntime.configuration import ORTConfig
from optimum.onnxruntime.utils import generate_identified_filename


logger = logging.getLogger(__name__)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class OnnxConfigManager:
    """A class that notes down the attribute names(for `number of heads` and `hidden size`) of models in `transformers/models`. 
    It is optional for BERT model, but for other model types, you need specify the name of these parameters. It is possible to add
    customized model information with `update_model()` method.
        Args:
            __conf (:obj:`dict`):
                Dictionary register the attribute names of models.
        """
    __conf = {
        "bert": {"num_heads":"num_attention_heads", "hidden_size":"hidden_size"},
        "distilbert": {"num_heads":"n_heads", "hidden_size":"hidden_size"},
        "roberta": {"num_heads":"num_attention_heads", "hidden_size":"hidden_size"},
        "bart": {"num_heads":"encoder_attention_heads", "hidden_size":"d_model"},
        "gpt2": {"num_heads":"n_head", "hidden_size":"n_embd"},
        }

    @staticmethod
    def get_num_heads(model_type:str) -> str:
        try:
            return OnnxConfigManager.__conf[model_type]["num_heads"]
        except KeyError:
            print(f"{model_type} undefined in the configuration, please define it with `add_model` or it will be set to default value.")
            return "num_heads"
    
    @staticmethod
    def get_hidden_size(model_type:str) -> str:
        try:
            return OnnxConfigManager.__conf[model_type]["hidden_size"]
        except KeyError:
            print(f"{model_type} undefined in the configuration, please define it with `add_model` or it will be set to default value.")
            return "hidden_size"
    
    @staticmethod
    def update_model(model_type:str, num_heads:str, hidden_size:str):
        OnnxConfigManager.__conf[model_type] = {"num_heads":num_heads, "hidden_size":hidden_size}

    @staticmethod
    def remove_model(model_type:str):
        try:
            OnnxConfigManager.__conf.pop(model_type)
        except KeyError:
            print(f"{model_type} undefined in the configuration")
    
    @staticmethod
    def check_supported_model(model_type:str) -> bool:
        has_model = model_type in OnnxConfigManager.__conf.keys()
        return has_model


class ORTOptimizer:
    def __init__(
        self,
        model_name_or_path: str,
        ort_config: Union[str, ORTConfig],
        feature: str = "default",
        **kwargs
    ):
        """
        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            ort_config (:obj:`Union[ORTConfig, str]`):
                Configuration file containing all the information related to the model optimization and quantization.
                Can be either:
                    - an instance of the class :class:`ORTConfig`,
                    - a string valid as input to :func:`ORTConfig.from_pretrained`.
            feature (:obj:`str`):
                Feature used when exporting the model.
            data_files (:obj:`str`, `optional`):
                Path to source data files.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(:obj:`str`, `optional`):
                The specific version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
        """
        config_kwargs_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None),
        ]
        config_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in config_kwargs_default}
        self.cache_dir = config_kwargs.get("cache_dir")
        self.model_name_or_path = model_name_or_path
        if not isinstance(ort_config, ORTConfig):
            ort_config = ORTConfig.from_pretrained(ort_config, **config_kwargs)
        self.ort_config = ort_config
        self.onnx_config = None
        self.feature = feature
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = FeaturesManager.get_model_from_feature(self.feature, self.model_name_or_path)
        self.onnx_model_path = None
        self.optim_model_path = None

    def export(self, model_path: os.PathLike) -> None:
        """
        Load and export a model to an ONNX Intermediate Representation (IR).

        Param:
            model_path (:obj:`os.PathLike`):
                The path used to save the model exported to an ONNX Intermediate Representation (IR).
        """

        model_type, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            self.model, feature=self.feature
        )
        self.onnx_config = model_onnx_config(self.model.config)
        opset = self.onnx_config.default_onnx_opset if self.ort_config.opset is None else self.ort_config.opset
        _ = export(self.tokenizer, self.model, self.onnx_config, opset, model_path)

    def fit(self, output_dir: Union[str, os.PathLike], **kwargs) -> None:
        """
        Load and export a model to an ONNX Intermediate Representation (IR) and apply the graph optimization.

        Param:
            output_dir (:obj:`Union[str, os.PathLike]`):
                The output directory where the optimized model will be saved.
        """
        output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
        self.onnx_model_path = output_dir.joinpath("model.onnx")
        self.optim_model_path = generate_identified_filename(self.onnx_model_path, "-optimized")

        self.export(self.onnx_model_path)
        config = self.onnx_config._config
        model_type = getattr(config, "model_type")
        onnx_config_defined = OnnxConfigManager.check_supported_model(model_type)
        num_heads = getattr(config, OnnxConfigManager.get_num_heads(model_type)) if onnx_config_defined else 0
        hidden_size =  getattr(config, OnnxConfigManager.get_hidden_size(model_type)) if onnx_config_defined else 0
        model_type = "bert" if "bert" in model_type else model_type

        fusion_config_kwargs_default = [
            ("model_type", model_type),
            ("disable_gelu", False),
            ("disable_layer_norm", False),
            ("disable_attention", False),
            ("disable_skip_layer_norm", False),
            ("disable_bias_skip_layer_norm", False),
            ("disable_bias_gelu", False),
            ("enable_gelu_approximation", False),
            ("use_mask_index", False),
            ("no_attention_mask", False),
            ("disable_embed_layer_norm", True),
        ]
        fusion_config_kwargs = AttrDict({name: kwargs.get(name, default_value) for (name, default_value) in fusion_config_kwargs_default})
        optimization_options = FusionOptions.parse(fusion_config_kwargs)

        print()
        optimizer = optimize_model(
            self.onnx_model_path.as_posix(),
            model_type,
            num_heads,
            hidden_size,
            opt_level=self.ort_config.opt_level,
            optimization_options=optimization_options,
            use_gpu=self.ort_config.use_gpu,
            only_onnxruntime=self.ort_config.only_onnxruntime,
        )

        optimizer.save_model_to_file(self.optim_model_path.as_posix(), self.ort_config.use_external_data_format)
        print(f"Optimized model saved to: {self.optim_model_path}")
        
        if optimizer.is_fully_optimized():
            print("The model has been fully optimized.")
        else:
            print("The model has been optimized.")

    def get_optimize_details(
        self, onnx_model_path: Optional[str] = None, optimized_model_path: Optional[str] = None, 
        summary:bool=True, nodes_details:bool=True) -> dict:
        """
        Returns a dictionary reporting the optimization.
        Param:
            onnx_model_path (:obj:`str`, `optional`):
                Path of a stored onnx model.
            optimized_model_path (:obj:`str`, `optional`):
                Path of the corresponding optimized onnx model.
            summary (:obj:`bool`):
                Whether report the optimization details: reduction of nodes, and complex node fusions.
            nodes_details (:obj:`bool`):
                Whether report the top 5 reduced op_types, and return the detailed node change list.
        Return:
            sorted_nodes_change (:obj:`dict`):
                Returns a sorted list with op types and its change after the optimization.
        """
        if self.onnx_model_path is None and onnx_model_path is None:
            raise AttributeError(
                "No `self.onnx_model_path` attribute, please input the value of `onnx_model_path` argument."
            )
        if self.optim_model_path is None and optimized_model_path is None:
            raise AttributeError(
                "No `self.optimized_model_path` attribute, please input the value of `optimized_model_path` argument."
            )
        onnx_model_path = onnx_model_path if onnx_model_path else self.onnx_model_path
        optimized_model_path = optimized_model_path if optimized_model_path else self.optim_model_path
        onnx_model = load_model(onnx_model_path, format=None, load_external_data=True)
        optim_model = load_model(optimized_model_path, format=None, load_external_data=True)
        onnx_model = BertOnnxModel(onnx_model)
        optimizer = BertOnnxModel(optim_model)
        
        def get_node_change(op_type):
            return len(onnx_model.get_nodes_by_op_type(op_type)) - len(optimizer.get_nodes_by_op_type(op_type))

        if summary:
            # Nodes reduction information
            count_nodes_onnx = len(onnx_model.nodes())
            count_nodes_optim = len(optimizer.nodes())
            print(f"There are {count_nodes_onnx} nodes before the optimization," 
                f"and {count_nodes_optim} nodes after the optimization."
                f" {count_nodes_onnx-count_nodes_optim} nodes are reduced.")
            # Extended fusion statistics
            extended_fusion_statistic = optimizer.get_fused_operator_statistics()
            print(f"Complex node fusions(if extended mode enabled, opt_level>1):\n{extended_fusion_statistic}")
        
        # Top 5 reduced operations & node details onnx model v.s. optimized model
        if nodes_details:
            op_types = []
            for model in [onnx_model, optimizer]:
                for node in model.nodes():
                    if node.op_type not in op_types: op_types.append(node.op_type)
                        
            nodes_change = dict(map(lambda op_type: (op_type, get_node_change(op_type)), op_types))
            sorted_nodes_change = sorted(nodes_change.items(), key=lambda op: abs(op[1]))
            sorted_nodes_change.reverse()
            print("Top 5 optimized ops are:\n", [op[0] for op in sorted_nodes_change[:5]])
            return sorted_nodes_change
