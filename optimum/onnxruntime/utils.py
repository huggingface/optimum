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

from packaging.version import parse
from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.file_utils import is_torch_onnx_dict_inputs_support_available
from transformers.modeling_utils import PreTrainedModel
from transformers.onnx.config import OnnxConfig
from transformers.onnx.convert import ensure_model_and_config_inputs_match
from transformers.utils import logging


logger = logging.get_logger(__name__)


def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


def export_static(
    tokenizer: PreTrainedTokenizer, model: PreTrainedModel, config: OnnxConfig, opset: int, output: Path
) -> Tuple[List[str], List[str]]:
    """
    Export a PyTorch backed pipeline to static ONNX Intermediate Representation (IR
    """
    if not is_torch_available():
        raise ImportError("Cannot convert because PyTorch is not installed. Please install torch first.")

    import torch
    from torch.onnx import export
    from transformers.file_utils import torch_version

    if not is_torch_onnx_dict_inputs_support_available():
        raise AssertionError(f"Unsupported PyTorch version, minimum required is 1.8.0, got: {torch_version}")

    logger.info(f"Using framework PyTorch: {torch.__version__}")
    with torch.no_grad():
        model.config.return_dict = True
        model.eval()

        # Check if we need to override certain configuration item
        if config.values_override is not None:
            logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
            for override_config_key, override_config_value in config.values_override.items():
                logger.info(f"\t- {override_config_key} -> {override_config_value}")
                setattr(model.config, override_config_key, override_config_value)

        # Ensure inputs match
        # TODO: Check when exporting QA we provide "is_pair=True"
        model_inputs = config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)
        inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
        onnx_outputs = list(config.outputs.keys())

        if not inputs_match:
            raise ValueError("Model and config inputs doesn't match")

        config.patch_ops()

        # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
        # so we check the torch version for backwards compatibility
        if parse(torch.__version__) <= parse("1.10.99"):
            # export can work with named args but the dict containing named args
            # has to be the last element of the args tuple.
            export(
                model,
                (model_inputs,),
                f=output.as_posix(),
                input_names=list(config.inputs.keys()),
                output_names=onnx_outputs,
                # dynamic_axes={name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())},
                do_constant_folding=True,
                use_external_data_format=config.use_external_data_format(model.num_parameters()),
                enable_onnx_checker=True,
                opset_version=opset,
            )
        else:
            export(
                model,
                (model_inputs,),
                f=output.as_posix(),
                input_names=list(config.inputs.keys()),
                output_names=onnx_outputs,
                # dynamic_axes={name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())},
                do_constant_folding=True,
                opset_version=opset,
            )

        config.restore_ops()

    return matched_inputs, onnx_outputs
