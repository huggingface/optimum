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
from typing import Callable, Dict, List, Optional, Tuple, Union

from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager
from transformers.onnx.utils import get_preprocessor

from onnx import load_model
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.transformers.optimizer import get_fusion_statistics, optimize_model

from .configuration import OptimizationConfig
from .utils import ORTConfigManager


LOGGER = logging.getLogger(__name__)


class ORTOptimizer:
    """
    Handles the ONNX Runtime optimization process for models shared on huggingface.co/models.
    """

    @staticmethod
    def from_pretrained(
        model_name_or_path: Union[str, os.PathLike], feature: str, opset: Optional[int] = None
    ) -> "ORTOptimizer":
        """
        Instantiate a `ORTOptimizer` from a pretrained pytorch model and preprocessor.

        Args:
            model_name_or_path (`Union[str, os.PathLike]`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            feature (`str`):
                Feature to use when exporting the model.
            opset (`int`, *optional*):
                ONNX opset version to export the model with.

        Returns:
            An instance of `ORTOptimizer`.
        """
        preprocessor = get_preprocessor(model_name_or_path)
        model_class = FeaturesManager.get_model_class_for_feature(feature)
        model = model_class.from_pretrained(model_name_or_path)

        return ORTOptimizer(preprocessor, model, feature, opset)

    def __init__(
        self,
        preprocessor: Union[AutoFeatureExtractor, AutoProcessor, AutoTokenizer],
        model: PreTrainedModel,
        feature: str = "default",
        opset: Optional[int] = None,
    ):
        """
        Args:
            preprocessor (`Union[AutoFeatureExtractor, AutoProcessor, AutoTokenizer]`):
                The preprocessor used to preprocess the data.
            model (`PreTrainedModel`):
                The model to optimize.
            feature (`str`, defaults to `"default"`):
                Feature to use when exporting the model.
            opset (`int`, *optional*):
                ONNX opset version to export the model with.
        """
        super().__init__()

        self.preprocessor = preprocessor
        self.model = model
        self.feature = feature
        self._model_type, onnx_config_factory = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
        self._onnx_config = onnx_config_factory(self.model.config)
        self.opset = self._onnx_config.default_onnx_opset if opset is None else opset

    def export(
        self,
        onnx_model_path: Union[str, os.PathLike],
        onnx_optimized_model_output_path: Union[str, os.PathLike],
        optimization_config: OptimizationConfig,
        use_external_data_format: bool = False,
        all_tensors_to_one_file: bool = True,
    ) -> Path:
        """
        Optimize a model given the optimization specifications defined in `optimization_config`.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                The path used to save the model exported to an ONNX Intermediate Representation (IR).
            onnx_optimized_model_output_path (`Union[str, os.PathLike]`):
                The path used to save the optimized model exported to an ONNX Intermediate Representation (IR).
            optimization_config (`OptimizationConfig`):
                The configuration containing the parameters related to optimization.
            use_external_data_format (`bool`, defaults to `False`):
                Whether to use external data format to store model whose size is >= 2Gb.
            all_tensors_to_one_file (`bool`, defaults to `True`):
                Whether to save all tensors to one external file specified by location. If false, save each tensor to a file named with the tensor name.

        Returns:
            The path of the resulting optimized model.
        """
        if not isinstance(onnx_model_path, Path):
            onnx_model_path = Path(onnx_model_path)

        if not isinstance(onnx_optimized_model_output_path, str):
            onnx_optimized_model_output_path = str(onnx_optimized_model_output_path)

        # Export the model if it has not already been exported to ONNX IR
        if not onnx_model_path.exists():
            export(self.preprocessor, self.model, self._onnx_config, self.opset, onnx_model_path)

        ORTConfigManager.check_supported_model_or_raise(self._model_type)
        num_heads = getattr(self.model.config, ORTConfigManager.get_num_heads_name(self._model_type))
        hidden_size = getattr(self.model.config, ORTConfigManager.get_hidden_size_name(self._model_type))
        model_type = ORTConfigManager.get_model_ort_type(self._model_type)
        optimization_config.model_type = model_type
        optimization_options = FusionOptions.parse(optimization_config)

        LOGGER.info(f"Creating optimizer: {optimization_config}")

        optimizer = optimize_model(
            onnx_model_path.as_posix(),
            model_type,
            num_heads,
            hidden_size,
            opt_level=optimization_config.optimization_level,
            optimization_options=optimization_options,
            use_gpu=optimization_config.optimize_for_gpu,
            only_onnxruntime=optimization_config.optimize_with_onnxruntime_only,
        )

        if optimization_config.fp16:
            # keep_io_types to keep inputs/outputs as float32
            optimizer.convert_float_to_float16(keep_io_types=True)

        optimizer.save_model_to_file(
            onnx_optimized_model_output_path, use_external_data_format, all_tensors_to_one_file
        )

        if optimizer.is_fully_optimized():
            msg = "The model has been fully optimized "
        else:
            msg = "The model has been optimized "

        LOGGER.info(
            msg + f"and saved at {onnx_optimized_model_output_path} (external data format: {use_external_data_format})"
        )

        return Path(onnx_optimized_model_output_path)

    @staticmethod
    def get_fused_operators(onnx_model_path: Union[str, os.PathLike]) -> Dict[str, int]:
        """
        Compute the dictionary mapping the name of the fused operators to their number of apparition in the model.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                Path of the ONNX model.

        Returns:
            The dictionary mapping the name of the fused operators to their number of apparition in the model.
        """
        onnx_optimized_model = BertOnnxModel(load_model(onnx_model_path))
        fused_operator = onnx_optimized_model.get_fused_operator_statistics()
        LOGGER.info(
            f"The following operators were fused : { ', '.join([k for k,v in fused_operator.items() if v > 0])}"
        )
        return {k: v for k, v in fused_operator.items() if v > 0}

    @staticmethod
    def get_nodes_number_difference(
        onnx_model_path: Union[str, os.PathLike], onnx_optimized_model_path: Union[str, os.PathLike]
    ) -> int:
        """
        Compute the difference in the number of nodes between the original and the optimized model.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                Path of the ONNX model.
            onnx_optimized_model_path (`Union[str, os.PathLike]`):
                Path of the optimized ONNX model.

        Returns:
            The difference in the number of nodes between the original and the optimized model.
        """
        onnx_model = BertOnnxModel(load_model(onnx_model_path))
        onnx_optimized_model = BertOnnxModel(load_model(onnx_optimized_model_path))

        # Information in the number of nodes decrease resulting from optimization
        nodes_number_onnx_model = len(onnx_model.nodes())
        nodes_number_onnx_optimized_model = len(onnx_optimized_model.nodes())
        difference_nodes_number = nodes_number_onnx_model - nodes_number_onnx_optimized_model
        LOGGER.info(
            f"There are {nodes_number_onnx_model} nodes before optimization and {nodes_number_onnx_optimized_model}"
            f"nodes after. The number of nodes removed is {difference_nodes_number}"
        )
        return difference_nodes_number

    @staticmethod
    def get_operators_difference(
        onnx_model_path: Union[str, os.PathLike], onnx_optimized_model_path: Union[str, os.PathLike]
    ) -> Dict[str, int]:
        """
        Compute the dictionary mapping the operators name to the difference in the number of corresponding nodes between
        the original and the optimized model.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                Path of the ONNX model.
            onnx_optimized_model_path (`Union[str, os.PathLike]`):
                Path of the optimized ONNX model.

        Returns:
            The dictionary mapping the operators name to the difference in the number of corresponding nodes between the
            original and the optimized model.
        """
        onnx_model = BertOnnxModel(load_model(onnx_model_path))
        onnx_optimized_model = BertOnnxModel(load_model(onnx_optimized_model_path))

        def nodes_difference_given_type(op_type):
            onnx_model_nodes_with_op_type = len(onnx_model.get_nodes_by_op_type(op_type))
            onnx_optimized_model_nodes_with_op_type = len(onnx_optimized_model.get_nodes_by_op_type(op_type))
            return onnx_model_nodes_with_op_type - onnx_optimized_model_nodes_with_op_type

        # Compute operators difference between the original and the optimized models
        op_types = set()
        for model in [onnx_model, onnx_optimized_model]:
            for node in model.nodes():
                op_types.add(node.op_type)

        operators_difference = dict(map(lambda op_type: (op_type, nodes_difference_given_type(op_type)), op_types))
        return {k: v for k, v in operators_difference.items() if v != 0}
