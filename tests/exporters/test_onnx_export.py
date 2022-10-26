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
from parameterized import parameterized

from transformers import AutoConfig, is_tf_available, is_torch_available
from transformers.testing_utils import require_onnx, require_tf, require_torch, require_vision, slow

from optimum.exporters.onnx import OnnxConfig, OnnxConfigWithPast, export, validate_model_outputs
from optimum.exporters.onnx.base import EXTERNAL_DATA_FORMAT_SIZE_LIMIT
from optimum.exporters.onnx.utils import ParameterFormat, compute_serialized_parameters_size


if is_torch_available() or is_tf_available():
    from optimum.exporters.features import FeaturesManager


PYTORCH_EXPORT_MODELS = {
    ("albert", "hf-internal-testing/tiny-albert"),
    ("bert", "bert-base-cased"),
    ("big-bird", "google/bigbird-roberta-base"),
    ("ibert", "kssteven/ibert-roberta-base"),
    ("camembert", "camembert-base"),
    ("clip", "openai/clip-vit-base-patch32"),
    ("convbert", "YituTech/conv-bert-base"),
    ("codegen", "Salesforce/codegen-350M-multi"),
    # ("deberta", "microsoft/deberta-base"),
    # ("deberta-v2", "microsoft/deberta-v2-xlarge"),
    ("convnext", "facebook/convnext-tiny-224"),
    ("detr", "facebook/detr-resnet-50"),
    ("distilbert", "distilbert-base-cased"),
    ("electra", "google/electra-base-generator"),
    ("resnet", "microsoft/resnet-50"),
    ("roberta", "roberta-base"),
    ("roformer", "junnyu/roformer_chinese_base"),
    ("squeezebert", "squeezebert/squeezebert-uncased"),
    ("mobilebert", "google/mobilebert-uncased"),
    ("mobilevit", "apple/mobilevit-small"),
    ("xlm", "xlm-clm-ende-1024"),
    ("xlm-roberta", "xlm-roberta-base"),
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
    ("perceiver", "deepmind/language-perceiver", ("masked-lm", "sequence-classification")),
    ("perceiver", "deepmind/vision-perceiver-conv", ("image-classification",)),
    # TODO: longformer
    # ("longformer", "allenai/longformer-base-4096"),
    ("yolos", "hustvl/yolos-tiny"),
    ("segformer", "nvidia/segformer-b0-finetuned-ade-512-512"),
    ("bloom", "bigscience/bloom-560m"),
    ("gpt2", "gpt2"),
    ("gptj", "anton-l/gpt-j-tiny-random"),
    ("gpt-neo", "EleutherAI/gpt-neo-125M"),
    ("bart", "facebook/bart-base"),
    ("mbart", "sshleifer/tiny-mbart"),
    ("t5", "t5-small"),
    ("marian", "Helsinki-NLP/opus-mt-en-de"),
    ("mt5", "google/mt5-small"),
    ("m2m-100", "facebook/m2m100_418M"),
    ("blenderbot-small", "facebook/blenderbot_small-90M"),
    ("blenderbot", "facebook/blenderbot-400M-distill"),
    ("bigbird-pegasus", "google/bigbird-pegasus-large-arxiv"),
    ("longt5", "google/long-t5-local-base"),
    # Disable for now as it causes fatal error `Floating point exception (core dumped)` and the subsequential tests are
    # not run.
    # ("longt5", "google/long-t5-tglobal-base"),
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

    # @require_torch
    # @patch("optimum.exporters.onnx.convert.is_torch_onnx_dict_inputs_support_available", return_value=False)
    # def test_ensure_pytorch_version_ge_1_8_0(self, mock_is_torch_onnx_dict_inputs_support_available):
    #     """
    #     Ensures we raise an Exception if the pytorch version is unsupported (< 1.8.0).
    #     """
    #     self.assertRaises(AssertionError, export, None, None, None, None, None)
    #     mock_is_torch_onnx_dict_inputs_support_available.assert_called()

    def test_compute_parameters_serialized_size(self):
        """
        This test ensures we compute a "correct" approximation of the underlying storage requirement (size) for all the
        parameters for the specified parameter's dtype.
        """
        self.assertEqual(compute_serialized_parameters_size(2, ParameterFormat.Float), 2 * ParameterFormat.Float.size)

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

    Default means no specific features is being enabled on the model.
    """

    @patch.multiple(OnnxConfig, __abstractmethods__=set())
    def test_use_external_data_format(self):
        """
        External data format is required only if the serialized size of the parameters if bigger than 2Gb.
        """
        TWO_GB_LIMIT = EXTERNAL_DATA_FORMAT_SIZE_LIMIT

        # No parameters
        self.assertFalse(OnnxConfig.use_external_data_format(0))

        # Some parameters
        self.assertFalse(OnnxConfig.use_external_data_format(1))

        # Almost 2Gb parameters
        self.assertFalse(OnnxConfig.use_external_data_format((TWO_GB_LIMIT - 1) // ParameterFormat.Float.size))

        # Exactly 2Gb parameters
        self.assertTrue(OnnxConfig.use_external_data_format(TWO_GB_LIMIT))

        # More than 2Gb parameters
        self.assertTrue(OnnxConfig.use_external_data_format((TWO_GB_LIMIT + 1) // ParameterFormat.Float.size))


class OnnxConfigWithPastTestCase(TestCase):
    """
    Cover the tests for model which have use_cache feature (i.e. "with_past" for ONNX)
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
        for name, model, *features in export_models_list:
            if features:
                feature_config_mapping = {
                    feature: FeaturesManager.get_config(name, "onnx", feature) for _ in features for feature in _
                }
                # feature_config_mapping = {
                #     feature: FeaturesManager.get_config(name, "onnx", feature) for feature in features
                # }

            else:
                feature_config_mapping = FeaturesManager.get_supported_features_for_model_type(name, "onnx")

            for feature, onnx_config_class_constructor in feature_config_mapping.items():
                models_to_test.append((f"{name}_{feature}", name, model, feature, onnx_config_class_constructor))
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

    def _onnx_export(self, test_name, name, model_name, feature, onnx_config_class_constructor, device="cpu"):
        model_class = FeaturesManager.get_model_class_for_feature(feature)
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)

        # Dynamic axes aren't supported for YOLO-like models. This means they cannot be exported to ONNX on CUDA devices.
        # See: https://github.com/ultralytics/yolov5/pull/8378
        if model.__class__.__name__.startswith("Yolos") and device != "cpu":
            return

        onnx_config = onnx_config_class_constructor(model.config)

        # We need to set this to some value to be able to test the outputs values for batch size > 1.
        if isinstance(onnx_config, OnnxConfigWithPast) and getattr(model.config, "pad_token_id", None) is None and feature == "sequence-classification":
            model.config.pad_token_id = 0

        # if is_torch_available():
        #     from transformers.utils import torch_version

        #     if torch_version < onnx_config.torch_onnx_minimum_version:
        #         pytest.skip(
        #             "Skipping due to incompatible PyTorch version. Minimum required is"
        #             f" {onnx_config.torch_onnx_minimum_version}, got: {torch_version}"
        #         )

        with NamedTemporaryFile("w") as output:
            try:
                onnx_inputs, onnx_outputs = export(
                    model, onnx_config, onnx_config.DEFAULT_ONNX_OPSET, Path(output.name), device=device
                )
                atol = onnx_config.ATOL_FOR_VALIDATION
                if isinstance(atol, dict):
                    atol = atol[feature.replace("-with-past", "")]
                validate_model_outputs(
                    onnx_config,
                    model,
                    Path(output.name),
                    onnx_outputs,
                    atol,
                )
            except (RuntimeError, ValueError) as e:
                self.fail(f"{name}, {feature} -> {e}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS))
    @slow
    @require_torch
    @require_vision
    def test_pytorch_export(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS))
    @slow
    @require_torch
    @require_vision
    def test_pytorch_export_on_cuda(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor, device="cuda")

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_MODELS))
    @slow
    @require_tf
    @require_vision
    def test_tensorflow_export(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor)
