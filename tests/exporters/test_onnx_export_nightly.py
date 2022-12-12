# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Optional
from unittest import TestCase
from unittest.mock import patch

import pytest
from transformers import AutoConfig, is_tf_available, is_torch_available
from transformers.testing_utils import require_onnx, require_tf, require_torch, require_vision, slow

from optimum.exporters.onnx import (
    OnnxConfig,
    OnnxConfigWithPast,
    export,
    export_models,
    get_decoder_models_for_export,
    get_encoder_decoder_models_for_export,
    validate_model_outputs,
    validate_models_outputs,
)
from optimum.utils.testing_utils import grid_parameters
from parameterized import parameterized


if is_torch_available() or is_tf_available():
    from optimum.exporters.tasks import TasksManager

PYTORCH_EXPORT_MODELS_TINY = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "beit": "microsoft/beit-base-patch16-224",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "bart": "facebook/bart-base",
    "big-bird": "hf-internal-testing/tiny-random-BigBirdModel",
    "bigbird-pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot-small": "facebook/blenderbot_small-90M",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "camembert-base",
    "clip": "hf-internal-testing/tiny-random-CLIPModel",
    "convbert": "hf-internal-testing/tiny-random-ConvBertModel",
    "codegen": "hf-internal-testing/tiny-random-CodeGenModel",
    "data2vec-text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec-vision": "facebook/data2vec-vision-base",
    "deberta": "hf-internal-testing/tiny-random-DebertaModel",
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "deit": "facebook/deit-small-patch16-224",
    "convnext": "facebook/convnext-tiny-224",
    "detr": "hf-internal-testing/tiny-random-detr",
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gpt2": "gpt2",
    "gpt-neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gptj": "anton-l/gpt-j-tiny-random",
    "groupvit": "nvidia/groupvit-gcc-yfcc",
    "ibert": "kssteven/ibert-roberta-base",
    "levit": "facebook/levit-128S",
    "layoutlm": "hf-internal-testing/tiny-random-LayoutLMModel",
    "layoutlmv3": "microsoft/layoutlmv3-base",
    "longt5": "hf-internal-testing/tiny-random-longt5",
    # "longformer": "allenai/longformer-base-4096",
    "m2m-100": "hf-internal-testing/tiny-random-m2m_100",
    "marian": "Helsinki-NLP/opus-mt-en-de",
    "mbart": "sshleifer/tiny-mbart",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    # "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    # "mobilenet_v2": "google/mobilenet_v2_0.35_96",
    "mobilevit": "apple/mobilevit-small",
    "mt5": "lewtun/tiny-random-mt5",
    "owlvit": "google/owlvit-base-patch32",
    "perceiver": "hf-internal-testing/tiny-random-PerceiverModel",
    # "rembert": "google/rembert",
    "resnet": "microsoft/resnet-50",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "segformer": "nvidia/segformer-b0-finetuned-ade-512-512",
    "squeezebert": "hf-internal-testing/tiny-random-SqueezeBertModel",
    "swin": "microsoft/swin-tiny-patch4-window7-224",
    "t5": "t5-small",
    "vit": "google/vit-base-patch16-224",
    "yolos": "hustvl/yolos-tiny",
    "whisper": "openai/whisper-tiny.en",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm-roberta": "hf-internal-testing/tiny-random-XLMRobertaXLModel",
}

VALIDATE_EXPORT_ON_SHAPES_SLOW = {
    "batch_size": [2, 4, 6],
    "sequence_length": [8, 17, 33, 64, 96, 154],
    "num_choices": [2, 4],
}

VALIDATE_EXPORT_ON_SHAPES_FAST = {
    "batch_size": [4],
    "sequence_length": [17],
    "num_choices": [4],
}


def _get_models_to_test(export_models_list):
    models_to_test = []
    if is_torch_available() or is_tf_available():
        for name, model, *tasks in export_models_list:
            if tasks:
                task_config_mapping = {
                    task: TasksManager.get_exporter_config_constructor(name, "onnx", task=task)
                    for _ in tasks
                    for task in _
                }
            else:
                task_config_mapping = TasksManager.get_supported_tasks_for_model_type(name, "onnx")

            for task, onnx_config_class_constructor in task_config_mapping.items():
                models_to_test.append((f"{name}_{task}", name, model, task, onnx_config_class_constructor))
        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy", "dummy", OnnxConfig.from_model_config)]


class OnnxExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _onnx_export(
        self,
        test_name,
        name,
        model_name,
        task,
        onnx_config_class_constructor,
        device="cpu",
        for_ort=False,
        validate_shapes: Optional[Dict] = None,
    ):
        model_class = TasksManager.get_model_class_for_task(task)
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)

        # Dynamic axes aren't supported for YOLO-like models. This means they cannot be exported to ONNX on CUDA devices.
        # See: https://github.com/ultralytics/yolov5/pull/8378
        if model.__class__.__name__.startswith("Yolos") and device != "cpu":
            return

        onnx_config = onnx_config_class_constructor(model.config)

        # We need to set this to some value to be able to test the outputs values for batch size > 1.
        if (
            isinstance(onnx_config, OnnxConfigWithPast)
            and getattr(model.config, "pad_token_id", None) is None
            and task == "sequence-classification"
        ):
            model.config.pad_token_id = 0

        if is_torch_available():
            from optimum.exporters.onnx.utils import TORCH_VERSION

            if not onnx_config.is_torch_support_available:
                pytest.skip(
                    "Skipping due to incompatible PyTorch version. Minimum required is"
                    f" {onnx_config.MIN_TORCH_VERSION}, got: {TORCH_VERSION}"
                )

        atol = onnx_config.ATOL_FOR_VALIDATION
        if isinstance(atol, dict):
            atol = atol[task.replace("-with-past", "")]

        if for_ort is True and (model.config.is_encoder_decoder or task.startswith("causal-lm")):
            fn_get_models_from_config = (
                get_encoder_decoder_models_for_export
                if model.config.is_encoder_decoder
                else get_decoder_models_for_export
            )

            with TemporaryDirectory() as tmpdirname:
                try:
                    onnx_inputs, onnx_outputs = export_models(
                        model,
                        onnx_config,
                        onnx_config.DEFAULT_ONNX_OPSET,
                        output_dir=Path(tmpdirname),
                        fn_get_models_from_config=fn_get_models_from_config,
                        device=device,
                    )

                    input_shapes_iterator = grid_parameters(input_shapes_grid, yield_dict=True)
                    for input_shapes in input_shapes_iterator:
                        validate_models_outputs(
                            onnx_config,
                            model,
                            onnx_outputs,
                            atol,
                            output_dir=Path(tmpdirname),
                            fn_get_models_from_config=fn_get_models_from_config,
                            input_shapes=input_shapes,
                        )
                except (RuntimeError, ValueError) as e:
                    self.fail(f"{name}, {task} -> {e}")
        else:
            with NamedTemporaryFile("w") as output:
                try:
                    onnx_inputs, onnx_outputs = export(
                        model, onnx_config, onnx_config.DEFAULT_ONNX_OPSET, Path(output.name), device=device
                    )

                    if validate_shapes is not None:
                        input_shapes_grid = validate_shapes
                    else:
                        input_shapes_grid = None

                    input_shapes_iterator = grid_parameters(input_shapes_grid, yield_dict=True)
                    for input_shapes in input_shapes_iterator:
                        validate_model_outputs(
                            onnx_config,
                            model,
                            Path(output.name),
                            onnx_outputs,
                            atol,
                            input_shapes=input_shapes,
                        )
                except (RuntimeError, ValueError) as e:
                    self.fail(f"{name}, {task} -> {e}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    def test_pytorch_export(
        self,
        test_name,
        name,
        model_name,
        task,
        onnx_config_class_constructor,
        validate_shapes=VALIDATE_EXPORT_ON_SHAPES_FAST,
    ):
        self._onnx_export(
            test_name, name, model_name, task, onnx_config_class_constructor, validate_shapes=validate_shapes
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @slow
    @require_torch
    @require_vision
    def test_pytorch_export_multiple_shapes(
        self,
        test_name,
        name,
        model_name,
        task,
        onnx_config_class_constructor,
        validate_shapes=VALIDATE_EXPORT_ON_SHAPES_SLOW,
    ):
        self._onnx_export(
            test_name, name, model_name, task, onnx_config_class_constructor, validate_shapes=validate_shapes
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    def test_pytorch_export_on_cuda(
        self,
        test_name,
        name,
        model_name,
        task,
        onnx_config_class_constructor,
        validate_shapes=VALIDATE_EXPORT_ON_SHAPES_FAST,
    ):
        self._onnx_export(
            test_name,
            name,
            model_name,
            task,
            onnx_config_class_constructor,
            device="cuda",
            validate_shapes=validate_shapes,
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @slow
    @require_torch
    @require_vision
    def test_pytorch_export_on_cuda_multiple_shapes(
        self,
        test_name,
        name,
        model_name,
        task,
        onnx_config_class_constructor,
        validate_shapes=VALIDATE_EXPORT_ON_SHAPES_SLOW,
    ):
        self._onnx_export(
            test_name,
            name,
            model_name,
            task,
            onnx_config_class_constructor,
            device="cuda",
            validate_shapes=validate_shapes,
        )
