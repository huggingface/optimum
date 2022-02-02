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
from typing import Callable, List, Optional, Tuple, Union

from transformers import AutoTokenizer, PretrainedConfig
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

    The required attribute names are for the number of attention heads `num_heads` and hidden size `hidden_size`.
    It is possible to add customized model information with the `update_model` method.

    Attributes:
        __conf (`dict`):
            The dictionary mapping each model type to a dictionary containing the model attribute names corresponding to
            the number of attention heads and hidden size.
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
        default = "num_attention_heads"
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
        logger.info(f"{model_type} is now defined in the configuration.")

    @staticmethod
    def check_supported_model(model_type: str) -> bool:
        return model_type in OnnxConfigManager.__conf


class ORTOptimizer:
    """
    ORTOptimizer is a class for ONNX Runtime optimization of models in `huggingface/transformers/models`.
    """

    def __init__(self, ort_config: Union[str, ORTConfig], **kwargs):
        """
        Args:
            ort_config (`Union[ORTConfig, str]`):
                Configuration file containing all the information related to the model optimization.
                Can be either:
                    - an instance of the class :class:`ORTConfig`,
                    - a string valid as input to :func:`ORTConfig.from_pretrained`.
            cache_dir (`str`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (`bool`, `optional`, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (`bool`, `optional`, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(`str`, `optional`):
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
        if not isinstance(ort_config, ORTConfig):
            ort_config = ORTConfig.from_pretrained(ort_config, **config_kwargs)
        self.ort_config = ort_config
        self.onnx_config = None
        self.onnx_model_path = None
        self.optim_model_path = None
        self.tokenizer = None
        self.model = None

    def export(
        self,
        model_name_or_path: Union[str, os.PathLike],
        output_path: Union[str, os.PathLike],
        feature: str = "default",
        **kwargs
    ) -> None:
        """
        Loads and exports a model to an ONNX Intermediate Representation (IR).

        Args:
            model_name_or_path (`Union[str, os.PathLike]`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            output_path (`os.PathLike`):
                The path used to save the model exported to an ONNX Intermediate Representation (IR).
            feature (`str`, defaults to `"default"`):
                Feature to use when exporting the model.
            cache_dir (`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, `optional`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, `optional`, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(`str`, `optional`):
                The specific version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
        """
        kwargs_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None),
        ]
        model_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in kwargs_default}
        tokenizer_kwargs = copy.deepcopy(model_kwargs)
        output_path = Path(output_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        model_class = FeaturesManager.get_model_class_for_feature(feature)
        self.model = model_class.from_pretrained(model_name_or_path, **model_kwargs)
        model_type, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self.model, feature=feature)
        self.onnx_config = model_onnx_config(self.model.config)
        opset = self.onnx_config.default_onnx_opset if self.ort_config.opset is None else self.ort_config.opset
        _ = export(self.tokenizer, self.model, self.onnx_config, opset, output_path)

    def fit(
        self,
        model_name_or_path: Union[str, os.PathLike],
        output_dir: Union[str, os.PathLike],
        feature: str = "default",
        config: Optional[PretrainedConfig] = None,
        **kwargs
    ) -> None:
        """
        Applies the ONNX Runtime graph-level optimization on a given model and saves the resulting model.

        Args:
            model_name_or_path (`Union[str, os.PathLike]`):
                Repository name in the Hugging Face Hub, path to a local directory hosting the model or path to a
                pre-existing onnx model.
            output_dir (`Union[str, os.PathLike]`):
                The output directory where the optimized model will be saved.
            feature (`str`, defaults to `"default"`):
                Feature to use when exporting the model.
            config (`PretrainedConfig`, `optional`):
                 A configuration associated to the pre-existing ONNX model.
            cache_dir (`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, `optional`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, `optional`, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(`str`, `optional`):
                The specific version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            disable_gelu (`bool`, `optional`, defaults to `False`):
                Whether or not to disable Gelu fusion.
            disable_layer_norm (`bool`, `optional`, defaults to `False`):
                Whether or not to disable LayerNormalization fusion.
            disable_attention (`bool`, `optional`, defaults to `False`):
                Whether or not to disable Attention fusion.
            disable_skip_layer_norm (`bool`, `optional`, defaults to `False`):
                Whether or not to disable SkipLayerNormalization fusion.
            disable_bias_skip_layer_norm (`bool`, `optional`, defaults to `False`):
                Whether or not to disable Add Bias and SkipLayerNormalization fusion.
            disable_bias_gelu (`bool`, `optional`, defaults to `False`):
                Whether or not to disable Add Bias and Gelu/FastGelu fusion.
            enable_gelu_approximation (`bool`, `optional`, defaults to `False`):
                Whether or not to enable Gelu/BiasGelu to FastGelu conversion. The default value
                is set to `False` since the approximation might slightly impact the accuracy of
                models.
            use_mask_index (`bool`, `optional`, defaults to `False`):
                Whether or not to use mask index instead of raw attention mask in attention operator.
            no_attention_mask (`bool`, `optional`, defaults to `False`):
                No attention mask. Only works for `model_type=bert`.
            disable_embed_layer_norm (`bool`, `optional`, defaults to `True`):
                Whether or not to disable EmbedLayerNormalization fusion. The default value is set to
                `True` since the fusion is incompatible with ONNX Runtime quantization.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_model_path = Path(model_name_or_path)
        if not self.onnx_model_path.is_file():
            self.onnx_model_path = output_dir.joinpath("model.onnx")
            self.export(model_name_or_path, self.onnx_model_path, feature=feature, **kwargs)
            config = self.model.config
        elif config is None:
            raise ValueError(
                "A configuration `config` associated to the model must be provided when a pre-existing ONNX model is "
                "provided."
            )
        self.optim_model_path = generate_identified_filename(self.onnx_model_path, "-optimized")
        model_type = getattr(config, "model_type")
        onnx_config_defined = OnnxConfigManager.check_supported_model(model_type)
        num_heads = getattr(config, OnnxConfigManager.get_num_heads(model_type)) if onnx_config_defined else 0
        hidden_size = getattr(config, OnnxConfigManager.get_hidden_size(model_type)) if onnx_config_defined else 0
        model_type = "bert" if "bert" in model_type else model_type

        optimization_kwargs_default = [
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
        optimization_kwargs = AttrDict(
            {name: kwargs.get(name, default_value) for (name, default_value) in optimization_kwargs_default}
        )
        optimization_options = FusionOptions.parse(optimization_kwargs)

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
            onnx_model_path (`str`, `optional`):
                Path of a stored ONNX model.
            optimized_model_path (`str`, `optional`):
                Path of the corresponding optimized ONNX model.
            summary (`bool`, defaults to `True`):
                Whether report the optimization details: reduction of nodes, and complex node fusions.
            nodes_details (`bool`, defaults to `True`):
                Whether report the top 5 reduced op_types, and return the detailed node change list.

        Returns:
            sorted_nodes_change (`List[Tuple[str, int]]`):
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
        onnx_model_path = onnx_model_path if onnx_model_path is not None else self.onnx_model_path
        optimized_model_path = optimized_model_path if optimized_model_path else self.optim_model_path
        onnx_model = load_model(onnx_model_path, format=None, load_external_data=True)
        optim_model = load_model(optimized_model_path, format=None, load_external_data=True)
        onnx_model = BertOnnxModel(onnx_model)
        optim_model = BertOnnxModel(optim_model)

        def get_node_change(op_type):
            return len(onnx_model.get_nodes_by_op_type(op_type)) - len(optim_model.get_nodes_by_op_type(op_type))

        if summary:
            # Nodes reduction information
            count_nodes_onnx = len(onnx_model.nodes())
            count_nodes_optim = len(optim_model.nodes())
            logger.info(
                f"There are {count_nodes_onnx} nodes before optimization and {count_nodes_optim} nodes after. "
                f"The number of nodes removed is {count_nodes_onnx - count_nodes_optim}."
            )
            if self.ort_config.opt_level and self.ort_config.opt_level > 1:
                # Extended fusion statistics
                extended_fusion_statistic = optim_model.get_fused_operator_statistics()
                logger.info("Complex node fusions:\n", extended_fusion_statistic)

        # Top 5 reduced operations & node details onnx model v.s. optimized model
        sorted_nodes_change = []
        if nodes_details:
            op_types = []
            for model in [onnx_model, optim_model]:
                for node in model.nodes():
                    if node.op_type not in op_types:
                        op_types.append(node.op_type)

            nodes_change = dict(map(lambda op_type: (op_type, get_node_change(op_type)), op_types))
            sorted_nodes_change = sorted(nodes_change.items(), key=lambda op: -abs(op[1]))
            logger.info("Top 5 optimized ops are:\n", [op[0] for op in sorted_nodes_change[:5]])
        return sorted_nodes_change
