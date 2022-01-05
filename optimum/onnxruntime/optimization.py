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

import copy
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Union

from transformers import AutoTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager
from transformers.utils import logging

import onnx
from onnx import load_model
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.transformers.optimizer import FusionOptions, get_fusion_statistics, optimize_model
from optimum.onnxruntime.configuration import ORTConfig
from optimum.onnxruntime.utils import generate_identified_filename


logger = logging.get_logger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class OnnxConfigManager:
    """
    A class that notes down the attribute names of models in `huggingface/transformers/models`.

    The required attribute names are for `number of heads` and `hidden size`. It is optional for
    BERT model, but for other model types, you need to specify the name of these parameters. It
    is possible to add customized model information with `update_model()` method.

    Attributes:
        __conf (:obj:`dict`):
            Dictionary registers the attribute names of models(number of heads and hidden size).
    """

    __conf = {
        "bert": {"num_heads": "num_attention_heads", "hidden_size": "hidden_size"},
        "distilbert": {"num_heads": "n_heads", "hidden_size": "hidden_size"},
        "roberta": {"num_heads": "num_attention_heads", "hidden_size": "hidden_size"},
        "bart": {"num_heads": "encoder_attention_heads", "hidden_size": "d_model"},
        "gpt2": {"num_heads": "n_head", "hidden_size": "n_embd"},
    }

    @staticmethod
    def get_num_heads(model_type: str) -> str:
        default = "num_heads"
        try:
            return OnnxConfigManager.__conf[model_type]["num_heads"]
        except KeyError:
            logger.warning(
                f"{model_type} undefined in the configuration, please define it with `update_model` or it will be set "
                f"to the default value {default}."
            )
            return default

    @staticmethod
    def get_hidden_size(model_type: str) -> str:
        default = "hidden_size"
        try:
            return OnnxConfigManager.__conf[model_type]["hidden_size"]
        except KeyError:
            logger.warning(
                f"{model_type} undefined in the configuration, please define it with `update_model` or it will be set "
                f"to the default value {default}."
            )
            return default

    @staticmethod
    def update_model(model_type: str, num_heads: str, hidden_size: str):
        OnnxConfigManager.__conf[model_type] = {"num_heads": num_heads, "hidden_size": hidden_size}

    @staticmethod
    def remove_model(model_type: str):
        try:
            OnnxConfigManager.__conf.pop(model_type)
        except KeyError:
            logger.warning(f"{model_type} undefined in the configuration.")

    @staticmethod
    def check_supported_model(model_type: str) -> bool:
        return model_type in OnnxConfigManager.__conf


class ORTOptimizer:
    """
    ORTOptimizer is a class for onnxruntime optimization of models in `huggingface/transformers/models`.

    ORTOptimzer allows exportation of onnx model(`export()`), the graph-level optimization with onnxruntime
    (`fit()`) and report of the optimization(`get_optimize_details()`).
    """

    def __init__(self, model_name_or_path: str, ort_config: Union[str, ORTConfig], feature: str = "default", **kwargs):
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
        model_kwargs = copy.deepcopy(config_kwargs)
        tokenizer_kwargs = copy.deepcopy(config_kwargs)
        self.model_name_or_path = model_name_or_path
        if not isinstance(ort_config, ORTConfig):
            ort_config = ORTConfig.from_pretrained(ort_config, **config_kwargs)
        self.ort_config = ort_config
        self.onnx_config = None
        self.feature = feature
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, **tokenizer_kwargs)
        model_class = FeaturesManager.get_model_class_for_feature(self.feature)
        self.model = model_class.from_pretrained(self.model_name_or_path, **model_kwargs)
        self.onnx_model_path = None
        self.optim_model_path = None

    def export(self, model_path: os.PathLike) -> None:
        """
        Exports a model to an ONNX Intermediate Representation (IR).

        Args:
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
        Exports a model to an ONNX Intermediate Representation (IR) and apply the graph-level optimization by
        onnxruntime.

        Args:
            output_dir (:obj:`Union[str, os.PathLike]`):
                The output directory where the optimized model will be saved.
            disable_gelu (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable Gelu fusion.
            disable_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable LayerNormalization fusion.
            disable_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable Attention fusion.
            disable_skip_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable SkipLayerNormalization fusion.
            disable_bias_skip_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable Add Bias and SkipLayerNormalization fusion.
            disable_bias_gelu (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable Add Bias and Gelu/FastGelu fusion.
            enable_gelu_approximation (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to enable Gelu/BiasGelu to FastGelu conversion. The default value
                is set to `False` since the approximation might slightly impact the accuracy of
                models.
            use_mask_index (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use mask index instead of raw attention mask in attention operator.
            no_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                No attention mask. Only works for `model_type=bert`.
            disable_embed_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to disable EmbedLayerNormalization fusion. The default value is set to
                `True` since the fusion is incompatible with onnxruntime quantization.
        """
        output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
        self.onnx_model_path = output_dir.joinpath("model.onnx")
        self.optim_model_path = generate_identified_filename(self.onnx_model_path, "-optimized")

        self.export(self.onnx_model_path)
        config = self.model.config
        model_type = getattr(config, "model_type")
        onnx_config_defined = OnnxConfigManager.check_supported_model(model_type)
        num_heads = getattr(config, OnnxConfigManager.get_num_heads(model_type)) if onnx_config_defined else 0
        hidden_size = getattr(config, OnnxConfigManager.get_hidden_size(model_type)) if onnx_config_defined else 0
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
        fusion_config_kwargs = AttrDict(
            {name: kwargs.get(name, default_value) for (name, default_value) in fusion_config_kwargs_default}
        )
        optimization_options = FusionOptions.parse(fusion_config_kwargs)

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

        if optimizer.is_fully_optimized():
            msg = "The model has been fully optimized"
        else:
            msg = "The model has been optimized"

        logger.info(msg + f" and saved at {self.optim_model_path}")

    def get_optimize_details(
        self,
        onnx_model_path: Optional[str] = None,
        optimized_model_path: Optional[str] = None,
        summary: bool = True,
        nodes_details: bool = True,
    ) -> dict:
        """
        Returns a dictionary reporting the optimization.

        Args:
            onnx_model_path (:obj:`str`, `optional`):
                Path of a stored onnx model.
            optimized_model_path (:obj:`str`, `optional`):
                Path of the corresponding optimized onnx model.
            summary (:obj:`bool`):
                Whether report the optimization details: reduction of nodes, and complex node fusions.
            nodes_details (:obj:`bool`):
                Whether report the top 5 reduced op_types, and return the detailed node change list.

        Returns:
            sorted_nodes_change (:obj: `dict`):
                Returns a sorted list with op types and its change after the optimization.
        """
        if self.onnx_model_path is None and onnx_model_path is None:
            raise ValueError(
                "ORTOptimizer: a path toward the original onnx model `onnx_model_path` is needed to get the "
                "optimization details."
            )

        if self.optim_model_path is None and optimized_model_path is None:
            raise ValueError(
                "ORTOptimizer: a path toward the optimized onnx model `optimized_model_path` is needed to get the "
                "optimization details."
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
            logger.info(
                f"There are {count_nodes_onnx} nodes before optimization and {count_nodes_optim} nodes after. "
                f"The number of nodes removed is {count_nodes_onnx - count_nodes_optim}."
            )
            # Extended fusion statistics
            extended_fusion_statistic = optimizer.get_fused_operator_statistics()
            logger.info("Complex node fusions(if extended mode enabled, opt_level>1):\n", extended_fusion_statistic)

        # Top 5 reduced operations & node details onnx model v.s. optimized model
        if nodes_details:
            op_types = []
            for model in [onnx_model, optimizer]:
                for node in model.nodes():
                    if node.op_type not in op_types:
                        op_types.append(node.op_type)

            nodes_change = dict(map(lambda op_type: (op_type, get_node_change(op_type)), op_types))
            sorted_nodes_change = sorted(nodes_change.items(), key=lambda op: -abs(op[1]))
            logger.info("Top 5 optimized ops are:\n", [op[0] for op in sorted_nodes_change[:5]])
            return sorted_nodes_change
