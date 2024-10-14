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
"""Main class for performing graph optimization with ONNX Runtime."""

import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import onnx
from onnx import load_model
from transformers import GenerationConfig
from transformers.models.auto.configuration_auto import AutoConfig

from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.transformers.optimizer import optimize_model

from ..onnx.utils import check_model_uses_external_data
from ..utils import CONFIG_NAME, NormalizedConfigManager, logging
from ..utils.save_utils import maybe_save_preprocessors
from .configuration import OptimizationConfig, ORTConfig
from .modeling_decoder import ORTModelForCausalLM
from .modeling_ort import ORTModel
from .modeling_seq2seq import ORTModelForConditionalGeneration
from .utils import ONNX_WEIGHTS_NAME, ORTConfigManager


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.get_logger()


class ORTOptimizer:
    """
    Handles the ONNX Runtime optimization process for models shared on huggingface.co/models.
    """

    def __init__(self, onnx_model_path: List[os.PathLike], config: "PretrainedConfig", from_ortmodel: bool = False):
        """
        Args:
            onnx_model_path (`List[os.PathLike]`):
                The paths of the onnx models to optimize.
            config ([`~transformers.PretrainedConfig`]):
                An instance of the configuration associated to the model to optimize.
            from_ortmodel (`bool`, defaults to `False`):
                Whether the model being optimized is already loaded into an ORTModel, or if it was passed from disk.
        """
        super().__init__()
        self.onnx_model_path = onnx_model_path
        self.config = config
        self.model_type = self.config.model_type
        self.from_ortmodel = from_ortmodel

        try:
            self.normalized_config = NormalizedConfigManager.get_normalized_config_class(self.model_type)(self.config)
        except KeyError:
            raise NotImplementedError(
                f"Tried to use ORTOptimizer for the model type {self.model_type}, but it is not available yet. Please open an issue"
                " or submit a PR at https://github.com/huggingface/optimum."
            )

    @classmethod
    def from_pretrained(
        cls, model_or_path: Union[str, os.PathLike, ORTModel], file_names: Optional[List[str]] = None
    ) -> "ORTOptimizer":
        """
        Args:
            model_or_path (`Union[str, os.PathLike, ORTModel]`):
                The path to a local directory hosting the model to optimize or an instance of an `ORTModel` to quantize.
                Can be either:
                    - A path to a local *directory* containing the model to optimize.
                    - An instance of [`~optimum.onnxruntime.ORTModel`].
            file_names(`Optional[List[str]]`, defaults to `None`):
                The list of file names of the models to optimize.
        """
        onnx_model_path = []
        config = None
        if isinstance(model_or_path, ORTModel):
            from_ortmodel = True
            if isinstance(model_or_path, ORTModelForConditionalGeneration):
                onnx_model_path += [
                    model_or_path.encoder_model_path,
                    model_or_path.decoder_model_path,
                ]
                # Add the decoder with past key/values if present
                if model_or_path.use_cache:
                    onnx_model_path.append(model_or_path.decoder_with_past_model_path)
            elif isinstance(model_or_path, ORTModelForCausalLM) and model_or_path.use_merged:
                raise NotImplementedError(
                    "ORTOptimizer does not support ORTModelForCausalLM models when without/with past models are merged. "
                    "Please re-export your model. This can be done by using the optimum-cli ONNX export tool or `ORTModelForCausalLM.from_pretrained(..., export=True, use_merged=False)`."
                )
            else:
                onnx_model_path.append(model_or_path.model_path)
            config = model_or_path.config
        elif os.path.isdir(model_or_path):
            from_ortmodel = False
            file_names = [ONNX_WEIGHTS_NAME] if file_names is None else file_names
            model_or_path = Path(model_or_path)
            if CONFIG_NAME not in os.listdir(model_or_path):
                raise ValueError(f"The local directory does not contain the configuration file {CONFIG_NAME}.")
            config = AutoConfig.from_pretrained(model_or_path)
            for file_name in file_names:
                onnx_model_path.append(model_or_path.joinpath(file_name))
        else:
            raise ValueError(f"Unable to load the model from {model_or_path}.")
        return cls(onnx_model_path, config=config, from_ortmodel=from_ortmodel)

    def optimize(
        self,
        optimization_config: OptimizationConfig,
        save_dir: Union[str, os.PathLike],
        file_suffix: Optional[str] = "optimized",
        use_external_data_format: Optional[bool] = None,
        one_external_file: bool = True,
    ):
        """
        Optimizes a model given the optimization specifications defined in `optimization_config`.

        Args:
            optimization_config ([`~optimum.onnxruntime.OptimizationConfig`]):
                The configuration containing the parameters related to optimization.
            save_dir (`Union[str, os.PathLike]`):
                The path used to save the optimized model.
            file_suffix (`str`, defaults to `"optimized"`):
                The file suffix used to save the optimized model.
            use_external_data_format (`Optional[bool]`, defaults to `None`):
                Whether to use external data format to store model of size >= 2Gb. This argument is deprecated.
            one_external_file (`bool`, defaults to `True`):
                When `use_external_data_format=True`, whether to save all tensors to one external file.
                If False, save each tensor to a file named with the tensor name.

        """
        if use_external_data_format is not None:
            logger.warning(
                "The argument use_external_data_format in the ORTOptimizer.optimize() method is deprecated and will"
                " be removed in optimum 2.0."
            )

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ORTConfigManager.check_optimization_supported_model(self.model_type, optimization_config)
        model_type = ORTConfigManager.get_model_ort_type(self.config.model_type)
        optimization_options = optimization_config.create_fusion_options(model_type)

        logger.info("Optimizing model...")

        # TODO: this is quite inefficient as we load in memory if models are <2GB without external data
        model_uses_external_data = False
        for model_path in self.onnx_model_path:
            # check if external data was exported
            onnx_model = onnx.load(str(model_path), load_external_data=False)
            if check_model_uses_external_data(onnx_model) is True:
                model_uses_external_data = True
                break
        del onnx_model
        gc.collect()

        # Create and save the configuration summarizing all the parameters related to optimization
        ort_config = ORTConfig(
            optimization=optimization_config,
            use_external_data_format=model_uses_external_data,
            one_external_file=one_external_file,
        )

        for model_path in self.onnx_model_path:
            suffix = f"_{file_suffix}" if file_suffix else ""
            output_path = save_dir.joinpath(f"{model_path.stem}{suffix}").with_suffix(model_path.suffix)

            try:
                optimizer = optimize_model(
                    model_path.as_posix(),
                    model_type,
                    self.normalized_config.num_attention_heads,
                    self.normalized_config.hidden_size,
                    opt_level=optimization_config.optimization_level,
                    optimization_options=optimization_options,
                    use_gpu=optimization_config.optimize_for_gpu,
                    only_onnxruntime=not optimization_config.enable_transformers_specific_optimizations,
                )

                if optimization_config.fp16:
                    if model_uses_external_data:
                        # Refer to https://github.com/microsoft/onnxruntime/blob/v1.15.0/onnxruntime/python/tools/transformers/float16.py#L204
                        # The ONNX infer_shapes_path method should be used instead of infer_shapes
                        # for models >= 2 GB, and it expects a model written to disk.
                        # Note that convert_float_to_float16 then overwrites optimizer.model as the
                        # new ModelProto.
                        optimizer.save_model_to_file(
                            output_path.as_posix(),
                            use_external_data_format=model_uses_external_data,
                            all_tensors_to_one_file=one_external_file,
                        )

                        optimizer.model = output_path.as_posix()

                    # keep_io_types to keep inputs/outputs as float32
                    optimizer.convert_float_to_float16(
                        use_symbolic_shape_infer=not optimization_config.disable_shape_inference, keep_io_types=True
                    )
            except Exception as e:
                if "Incomplete symbolic shape inference" in str(e):
                    err = RuntimeError(
                        f"{str(e)}. Try to set `disable_shape_inference=True` in your optimization configuration."
                    )
                    raise err from e
                raise

            # TODO: ORT save_model_to_file will save as `.data` although we save as `.onnx_data` in the export
            optimizer.save_model_to_file(
                output_path.as_posix(),
                use_external_data_format=model_uses_external_data,
                all_tensors_to_one_file=one_external_file,
            )

            # if loading from disk and saving in the same repository, remove previous external data
            if Path(model_path.as_posix() + "_data").is_file() and self.from_ortmodel is False:
                os.remove(model_path.as_posix() + "_data")

        # Save the model configuration
        self.config.save_pretrained(save_dir)
        ort_config.save_pretrained(save_dir)
        maybe_save_preprocessors(self.onnx_model_path[0].parent, save_dir)

        try:
            generation_config = GenerationConfig.from_pretrained(self.onnx_model_path[0].parent)
            generation_config.save_pretrained(save_dir)
        except Exception:
            pass

        logger.info(
            f"Optimized model saved at: {save_dir} (external data format: "
            f"{model_uses_external_data}; saved all tensor to one file: "
            f"{one_external_file})"
        )

        return Path(save_dir)

    @staticmethod
    def get_fused_operators(onnx_model_path: Union[str, os.PathLike]) -> Dict[str, int]:
        """
        Computes the dictionary mapping the name of the fused operators to their number of apparition in the model.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                Path of the ONNX model.

        Returns:
            The dictionary mapping the name of the fused operators to their number of apparition in the model.
        """
        onnx_optimized_model = BertOnnxModel(load_model(onnx_model_path))
        fused_operator = onnx_optimized_model.get_fused_operator_statistics()
        logger.info(
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
        logger.info(
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

        operators_difference = {op_type: nodes_difference_given_type(op_type) for op_type in op_types}
        return {k: v for k, v in operators_difference.items() if v != 0}
