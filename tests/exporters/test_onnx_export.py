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
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest.mock import patch

import pytest
from transformers import AutoConfig, is_tf_available, is_torch_available
from transformers.testing_utils import require_onnx, require_tf, require_torch, require_vision, slow

from optimum.exporters.onnx import OnnxConfig, OnnxConfigWithPast, export, validate_model_outputs
from parameterized import parameterized


if is_torch_available() or is_tf_available():
    from optimum.exporters.tasks import TasksManager


PYTORCH_EXPORT_MODELS = {
    ("albert", "albert-base-v2"),
    ("bert", "bert-base-cased"),
    ("big-bird", "google/bigbird-roberta-base"),
    ("ibert", "kssteven/ibert-roberta-base"),
    ("camembert", "camembert-base"),
    ("clip", "openai/clip-vit-base-patch32"),
    ("convbert", "YituTech/conv-bert-base"),
    # Not using Salesforce/codegen-350M-multi because it takes too much time for testing.
    ("codegen", "hf-internal-testing/tiny-random-codegen"),
    # Not using microsoft/deberta-base because it takes too much time for testing.
    ("deberta", "hf-internal-testing/tiny-random-deberta"),
    # Not using microsoft/deberta-v2-xlarge because it takes too much time for testing.
    ("deberta-v2", "hf-internal-testing/tiny-random-deberta-v2"),
    ("convnext", "facebook/convnext-tiny-224"),
    # Not using facebook/detr-resnet-50 because it takes too much time for testing.
    ("detr", "hf-internal-testing/tiny-random-detr"),
    ("distilbert", "distilbert-base-cased"),
    ("electra", "google/electra-base-generator"),
    ("resnet", "microsoft/resnet-50"),
    ("roberta", "roberta-base"),
    ("roformer", "junnyu/roformer_chinese_base"),
    ("squeezebert", "squeezebert/squeezebert-uncased"),
    ("mobilebert", "google/mobilebert-uncased"),
    ("mobilevit", "apple/mobilevit-small"),
    ("xlm", "xlm-clm-ende-1024"),
    # Not using xlm-roberta-base because it takes too much time for testing.
    ("xlm-roberta", "Unbabel/xlm-roberta-comet-small"),
    ("layoutlm", "microsoft/layoutlm-base-uncased"),
    # ("layoutlmv2", "microsoft/layoutlmv2-base-uncased"),
    ("layoutlmv3", "microsoft/layoutlmv3-base"),
    ("groupvit", "nvidia/groupvit-gcc-yfcc"),
    ("levit", "facebook/levit-128S"),
    ("owlvit", "google/owlvit-base-patch32"),
    ("vit", "google/vit-base-patch16-224"),
    ("deit", "facebook/deit-small-patch16-224"),
    ("beit", "microsoft/beit-base-patch16-224"),
    ("data2vec-text", "facebook/data2vec-text-base"),
    ("data2vec-vision", "facebook/data2vec-vision-base"),
    # Not using deepmind/language-perceiver because it takes too much time for testing.
    ("perceiver", "hf-internal-testing/tiny-random-language_perceiver", ("masked-lm", "sequence-classification")),
    # Not using deepmind/vision-perceiver-conv because it takes too much time for testing.
    ("perceiver", "hf-internal-testing/tiny-random-vision_perceiver_conv", ("image-classification",)),
    # TODO: longformer
    # ("longformer", "allenai/longformer-base-4096"),
    ("yolos", "hustvl/yolos-tiny"),
    ("segformer", "nvidia/segformer-b0-finetuned-ade-512-512"),
    # Not using bigscience/bloom-560m because it goes OOM.
    ("bloom", "hf-internal-testing/tiny-random-bloom"),
    ("gpt2", "gpt2"),
    ("gptj", "anton-l/gpt-j-tiny-random"),
    ("gpt-neo", "EleutherAI/gpt-neo-125M"),
    ("bart", "facebook/bart-base"),
    ("mbart", "sshleifer/tiny-mbart"),
    ("t5", "t5-small"),
    ("marian", "Helsinki-NLP/opus-mt-en-de"),
    # Not using google/mt5-small because it takes too much time for testing.
    ("mt5", "lewtun/tiny-random-mt5"),
    # Not using facebook/m2m100_418M because it takes too much time for testing.
    ("m2m-100", "hf-internal-testing/tiny-random-m2m_100"),
    ("blenderbot-small", "facebook/blenderbot_small-90M"),
    # Not using facebook/blenderbot-400M-distill because it takes too much time for testing, and might cause OOM.
    ("blenderbot", "facebook/blenderbot-90M"),
    # Not using google/bigbird-pegasus-large-arxiv because it takes too much time for testing.
    ("bigbird-pegasus", "hf-internal-testing/tiny-random-bigbird_pegasus"),
    # Not using google/long-t5-local-base because it takes too much time for testing.
    ("longt5", "hf-internal-testing/tiny-random-longt5"),
}


TENSORFLOW_EXPORT_MODELS = {
    ("albert", "hf-internal-testing/tiny-albert"),
    ("bert", "bert-base-cased"),
    ("camembert", "camembert-base"),
    ("distilbert", "distilbert-base-cased"),
    ("roberta", "roberta-base"),
}


@require_onnx
class OnnxUtilsTestCase(TestCase):
    """
    Covers all the utilities involved to export ONNX models.
    """

    def test_flatten_output_collection_property(self):
        """
        This test ensures we correctly flatten nested collection such as the one we use when returning past_keys.
        past_keys = Tuple[Tuple]

        ONNX exporter will export nested collections as ${collection_name}.${level_idx_0}.${level_idx_1}...${idx_n}
        """
        self.assertEqual(
            OnnxConfig.flatten_output_collection_property("past_key", [[0], [1], [2]]),
            {
                "past_key.0": 0,
                "past_key.1": 1,
                "past_key.2": 2,
            },
        )


class OnnxConfigTestCase(TestCase):
    """
    Covers the test for models default.

    Default means no specific tasks is being enabled on the model.
    """

    # TODO: insert relevant tests here.


class OnnxConfigWithPastTestCase(TestCase):
    """
    Cover the tests for model which have use_cache task (i.e. "with_past" for ONNX)
    """

    SUPPORTED_WITH_PAST_CONFIGS = ()

    @patch.multiple(OnnxConfigWithPast, __abstractmethods__=set())
    def test_use_past(self):
        """
        Ensures the use_past variable is correctly being set.
        """
        for name, config in OnnxConfigWithPastTestCase.SUPPORTED_WITH_PAST_CONFIGS:
            with self.subTest(name):
                self.assertFalse(
                    OnnxConfigWithPast.from_model_config(config()).use_past,
                    "OnnxConfigWithPast.from_model_config() should not use_past",
                )

                self.assertTrue(
                    OnnxConfigWithPast.with_past(config()).use_past,
                    "OnnxConfigWithPast.from_model_config() should use_past",
                )

    @patch.multiple(OnnxConfigWithPast, __abstractmethods__=set())
    def test_values_override(self):
        """
        Ensures the use_past variable correctly set the `use_cache` value in model's configuration.
        """
        for name, config in OnnxConfigWithPastTestCase.SUPPORTED_WITH_PAST_CONFIGS:
            with self.subTest(name):

                # Without past
                onnx_config_default = OnnxConfigWithPast.from_model_config(config())
                self.assertIsNotNone(onnx_config_default.values_override, "values_override should not be None")
                self.assertIn("use_cache", onnx_config_default.values_override, "use_cache should be present")
                self.assertFalse(
                    onnx_config_default.values_override["use_cache"], "use_cache should be False if not using past"
                )

                # With past
                onnx_config_default = OnnxConfigWithPast.with_past(config())
                self.assertIsNotNone(onnx_config_default.values_override, "values_override should not be None")
                self.assertIn("use_cache", onnx_config_default.values_override, "use_cache should be present")
                self.assertTrue(
                    onnx_config_default.values_override["use_cache"], "use_cache should be False if not using past"
                )


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

    def _onnx_export(self, test_name, name, model_name, task, onnx_config_class_constructor, device="cpu"):
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

        with NamedTemporaryFile("w") as output:
            try:
                onnx_inputs, onnx_outputs = export(
                    model, onnx_config, onnx_config.DEFAULT_ONNX_OPSET, Path(output.name), device=device
                )
                atol = onnx_config.ATOL_FOR_VALIDATION
                if isinstance(atol, dict):
                    atol = atol[task.replace("-with-past", "")]
                validate_model_outputs(
                    onnx_config,
                    model,
                    Path(output.name),
                    onnx_outputs,
                    atol,
                )
            except (RuntimeError, ValueError) as e:
                self.fail(f"{name}, {task} -> {e}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS))
    @slow
    @require_torch
    @require_vision
    def test_pytorch_export(self, test_name, name, model_name, task, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, task, onnx_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS))
    @slow
    @require_torch
    @require_vision
    def test_pytorch_export_on_cuda(self, test_name, name, model_name, task, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, task, onnx_config_class_constructor, device="cuda")

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_MODELS))
    @slow
    @require_tf
    @require_vision
    def test_tensorflow_export(self, test_name, name, model_name, task, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, task, onnx_config_class_constructor)
