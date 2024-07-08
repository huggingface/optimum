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
import gc
import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from unittest import TestCase

import onnx
import pytest
from parameterized import parameterized
from transformers import AutoConfig, is_tf_available, is_torch_available
from transformers.testing_utils import require_onnx, require_tf, require_torch, require_torch_gpu, require_vision, slow

from optimum.exporters.error_utils import AtolError
from optimum.exporters.onnx import (
    OnnxConfig,
    OnnxConfigWithPast,
    export_models,
    get_decoder_models_for_export,
    get_diffusion_models_for_export,
    get_encoder_decoder_models_for_export,
    main_export,
    onnx_export_from_model,
    validate_models_outputs,
)
from optimum.exporters.onnx.base import ConfigBehavior
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.exporters.onnx.constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED
from optimum.exporters.onnx.model_configs import WhisperOnnxConfig
from optimum.exporters.onnx.utils import get_speecht5_models_for_export
from optimum.utils import ONNX_WEIGHTS_NAME, DummyPastKeyValuesGenerator, NormalizedTextConfig
from optimum.utils.save_utils import maybe_load_preprocessors
from optimum.utils.testing_utils import grid_parameters, require_diffusers

from ..exporters_utils import (
    PYTORCH_DIFFUSION_MODEL,
    PYTORCH_EXPORT_MODELS_TINY,
    PYTORCH_SENTENCE_TRANSFORMERS_MODEL,
    PYTORCH_TIMM_MODEL,
    TENSORFLOW_EXPORT_MODELS,
    VALIDATE_EXPORT_ON_SHAPES_SLOW,
)


if is_torch_available() or is_tf_available():
    from optimum.exporters.tasks import TasksManager


SEED = 42


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


def _get_models_to_test(export_models_dict: Dict, library_name: str = "transformers"):
    models_to_test = []
    if is_torch_available() or is_tf_available():
        for model_type, model_names_tasks in export_models_dict.items():
            model_type = model_type.replace("_", "-")
            task_config_mapping = TasksManager.get_supported_tasks_for_model_type(
                model_type, "onnx", library_name=library_name
            )

            if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
                tasks = list(task_config_mapping.keys())
                model_tasks = {model_names_tasks: tasks}
            else:
                unique_tasks = set()
                for tasks in model_names_tasks.values():
                    for task in tasks:
                        unique_tasks.add(task)
                n_tested_tasks = len(unique_tasks)
                if n_tested_tasks != len(task_config_mapping):
                    raise ValueError(f"Not all tasks are tested for {model_type}.")
                model_tasks = model_names_tasks  # possibly, test different tasks on different models

            for model_name, tasks in model_tasks.items():
                for task in tasks:
                    if model_type == "encoder-decoder" and task == "seq2seq-lm-with-past":
                        # The model uses bert as decoder and does not support past key values
                        continue

                    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                        model_type=model_type,
                        exporter="onnx",
                        task=task,
                        model_name=model_name,
                        library_name=library_name,
                    )

                    models_to_test.append(
                        (f"{model_type}_{task}", model_type, model_name, task, onnx_config_constructor, False)
                    )

                    if any(
                        task == ort_special_task
                        for ort_special_task in [
                            "text-generation",
                            "text2text-generation",
                            "automatic-speech-recognition",
                            "image-to-text",
                        ]
                    ):
                        models_to_test.append(
                            (
                                f"{model_type}_{task}_monolith",
                                model_type,
                                model_name,
                                task,
                                onnx_config_constructor,
                                True,
                            )
                        )
        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy", "dummy", OnnxConfig)]


class OnnxExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _onnx_export(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
        onnx_config_class_constructor,
        shapes_to_validate: Dict,
        monolith: bool,
        device="cpu",
    ):
        library_name = TasksManager.infer_library_from_model(model_name)

        if library_name == "timm":
            model_class = TasksManager.get_model_class_for_task(task, library=library_name)
            model = model_class(f"hf_hub:{model_name}", pretrained=True, exportable=True)
            TasksManager.standardize_model_attributes(model, library_name=library_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            model_class = TasksManager.get_model_class_for_task(task, model_type=config.model_type.replace("_", "-"))
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
            and task == "text-classification"
        ):
            model.config.pad_token_id = 0

        if is_torch_available():
            from optimum.utils import torch_version

            if not onnx_config.is_transformers_support_available:
                import transformers

                pytest.skip(
                    "Skipping due to incompatible Transformers version. Minimum required is"
                    f" {onnx_config.MIN_TRANSFORMERS_VERSION}, got: {transformers.__version__}"
                )

            if not onnx_config.is_torch_support_available:
                pytest.skip(
                    "Skipping due to incompatible PyTorch version. Minimum required is"
                    f" {onnx_config.MIN_TORCH_VERSION}, got: {torch_version}"
                )

        atol = onnx_config.ATOL_FOR_VALIDATION
        if isinstance(atol, dict):
            atol = atol[task.replace("-with-past", "")]

        model_kwargs = None
        if (
            model.config.is_encoder_decoder
            and task.startswith(
                (
                    "text2text-generation",
                    "automatic-speech-recognition",
                    "image-to-text",
                    "feature-extraction-with-past",
                )
            )
            and monolith is False
        ):
            models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
        elif task.startswith("text-generation") and monolith is False:
            models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)
        elif model.config.model_type == "speecht5":
            model_kwargs = {"vocoder": "fxmarty/speecht5-hifigan-tiny"}
            models_and_onnx_configs = get_speecht5_models_for_export(model, onnx_config, model_kwargs)
        else:
            models_and_onnx_configs = {"model": (model, onnx_config)}

        with TemporaryDirectory() as tmpdirname:
            onnx_inputs, onnx_outputs = export_models(
                models_and_onnx_configs=models_and_onnx_configs,
                opset=onnx_config.DEFAULT_ONNX_OPSET,
                output_dir=Path(tmpdirname),
                device=device,
                model_kwargs=model_kwargs,
            )
            input_shapes_iterator = grid_parameters(shapes_to_validate, yield_dict=True, add_test_name=False)
            for input_shapes in input_shapes_iterator:
                skip = False
                for _, model_onnx_conf in models_and_onnx_configs.items():
                    if (
                        hasattr(model_onnx_conf[0].config, "max_position_embeddings")
                        and input_shapes["sequence_length"] >= model_onnx_conf[0].config.max_position_embeddings
                    ):
                        skip = True
                        break
                    if (
                        model_type == "groupvit"
                        and input_shapes["sequence_length"]
                        >= model_onnx_conf[0].config.text_config.max_position_embeddings
                    ):
                        skip = True
                        break
                if skip:
                    continue

                try:
                    validate_models_outputs(
                        models_and_onnx_configs=models_and_onnx_configs,
                        onnx_named_outputs=onnx_outputs,
                        atol=atol,
                        output_dir=Path(tmpdirname),
                        input_shapes=input_shapes,
                        device=device,
                        model_kwargs=model_kwargs,
                    )
                except AtolError as e:
                    print(f"The ONNX export succeeded with the warning: {e}")

                gc.collect()

    def _onnx_export_sd(self, model_type: str, model_name: str, device="cpu"):
        pipeline = TasksManager.get_model_from_task(model_type, model_name, device=device)
        models_and_onnx_configs = get_diffusion_models_for_export(pipeline)
        output_names = [os.path.join(name_dir, ONNX_WEIGHTS_NAME) for name_dir in models_and_onnx_configs]
        model, _ = models_and_onnx_configs["vae_encoder"]
        model.forward = lambda sample: {"latent_sample": model.encode(x=sample)["latent_dist"].parameters}

        with TemporaryDirectory() as tmpdirname:
            _, onnx_outputs = export_models(
                models_and_onnx_configs=models_and_onnx_configs,
                opset=14,
                output_dir=Path(tmpdirname),
                output_names=output_names,
                device=device,
            )
            validate_models_outputs(
                models_and_onnx_configs=models_and_onnx_configs,
                onnx_named_outputs=onnx_outputs,
                output_dir=Path(tmpdirname),
                atol=1e-3,
                onnx_files_subpaths=output_names,
                use_subprocess=False,
            )

    def test_all_models_tested(self):
        # make sure we test all models
        missing_models_set = (
            TasksManager._SUPPORTED_CLI_MODEL_TYPE
            - set(PYTORCH_EXPORT_MODELS_TINY.keys())
            - set(PYTORCH_TIMM_MODEL.keys())
            - set(PYTORCH_SENTENCE_TRANSFORMERS_MODEL.keys())
        )
        if len(missing_models_set) > 0:
            self.fail(f"Not testing all models. Missing models: {missing_models_set}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    @pytest.mark.run_slow
    @slow
    def test_pytorch_export_on_cpu(
        self,
        test_name,
        model_type,
        model_name,
        task,
        onnx_config_class_constructor,
        monolith: bool,
    ):
        if model_type == "speecht5" and monolith:
            self.skipTest("unsupported export")

        self._onnx_export(
            test_name,
            model_type,
            model_name,
            task,
            onnx_config_class_constructor,
            shapes_to_validate=VALIDATE_EXPORT_ON_SHAPES_SLOW,
            monolith=monolith,
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    @require_torch_gpu
    @slow
    @pytest.mark.run_slow
    @pytest.mark.gpu_test
    def test_pytorch_export_on_cuda(
        self,
        test_name,
        model_type,
        model_name,
        task,
        onnx_config_class_constructor,
        monolith: bool,
    ):
        if model_type == "speecht5" and monolith:
            self.skipTest("unsupported export")

        self._onnx_export(
            test_name,
            model_type,
            model_name,
            task,
            onnx_config_class_constructor,
            device="cuda",
            shapes_to_validate=VALIDATE_EXPORT_ON_SHAPES_SLOW,
            monolith=monolith,
        )

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_MODELS))
    @slow
    @pytest.mark.run_slow
    @require_tf
    @require_vision
    @pytest.mark.tensorflow_test
    def test_tensorflow_export(
        self, test_name, model_type, model_name, task, onnx_config_class_constructor, monolith: bool
    ):
        if monolith is False:
            return 0

        self._onnx_export(test_name, model_type, model_name, task, onnx_config_class_constructor, monolith=monolith)

    @parameterized.expand(PYTORCH_DIFFUSION_MODEL.items())
    @require_torch
    @require_vision
    @require_diffusers
    def test_pytorch_export_for_diffusion_models(self, model_type, model_name):
        self._onnx_export_sd(model_type, model_name)

    @parameterized.expand(PYTORCH_DIFFUSION_MODEL.items())
    @require_torch
    @require_vision
    @require_diffusers
    @require_torch_gpu
    @slow
    @pytest.mark.run_slow
    @pytest.mark.gpu_test
    def test_pytorch_export_for_diffusion_models_cuda(self, model_type, model_name):
        self._onnx_export_sd(model_type, model_name, device="cuda")


class CustomWhisperOnnxConfig(WhisperOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs

        if self._behavior is ConfigBehavior.ENCODER:
            for i in range(self._config.encoder_layers):
                common_outputs[f"encoder_attentions.{i}"] = {0: "batch_size"}
        elif self._behavior is ConfigBehavior.DECODER:
            for i in range(self._config.decoder_layers):
                common_outputs[f"decoder_attentions.{i}"] = {0: "batch_size", 3: "decoder_sequence_length"}
            for i in range(self._config.decoder_layers):
                common_outputs[f"cross_attentions.{i}"] = {0: "batch_size", 3: "cross_attention_length"}

        return common_outputs

    @property
    def torch_to_onnx_output_map(self):
        if self._behavior is ConfigBehavior.ENCODER:
            # The encoder export uses WhisperEncoder that returns the key "attentions"
            return {"attentions": "encoder_attentions"}
        else:
            return {}


class MPTDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    MPT swaps the two last dimensions for the key cache compared to usual transformers
    decoder models, thus the redefinition here.
    """

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.sequence_length,
        )
        past_value_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class CustomMPTOnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MPTDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MPTDummyPastKeyValuesGenerator

    DEFAULT_ONNX_OPSET = 14  # aten::tril operator requires opset>=14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        hidden_size="d_model", num_layers="n_layers", num_attention_heads="n_heads"
    )

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        """
        Adapted from https://github.com/huggingface/optimum/blob/v1.9.0/optimum/exporters/onnx/base.py#L625
        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 3: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 2: decoder_sequence_name}


def fn_get_submodels_custom(model, legacy=False):
    return {"decoder_model": model, "decoder_with_past_model": model} if legacy else {"model": model}


class OnnxCustomExport(TestCase):
    def test_custom_export_official_model(self):
        model_id = "openai/whisper-tiny.en"
        config = AutoConfig.from_pretrained(model_id)

        custom_whisper_onnx_config = CustomWhisperOnnxConfig(
            config=config,
            task="automatic-speech-recognition",
        )

        encoder_config = custom_whisper_onnx_config.with_behavior("encoder")
        decoder_config = custom_whisper_onnx_config.with_behavior("decoder", use_past=False)
        decoder_with_past_config = custom_whisper_onnx_config.with_behavior("decoder", use_past=True)

        custom_onnx_configs = {
            "encoder_model": encoder_config,
            "decoder_model": decoder_config,
            "decoder_with_past_model": decoder_with_past_config,
        }

        with TemporaryDirectory() as tmpdirname:
            main_export(
                model_id,
                output=tmpdirname,
                no_post_process=True,
                model_kwargs={"output_attentions": True},
                custom_onnx_configs=custom_onnx_configs,
            )

            model = onnx.load(os.path.join(tmpdirname, "decoder_model.onnx"))

            output_names = [outp.name for outp in model.graph.output]
            assert "decoder_attentions.0" in output_names
            assert "cross_attentions.0" in output_names

    @parameterized.expand([(None,), (fn_get_submodels_custom,)])
    def test_custom_export_trust_remote(self, fn_get_submodels):
        model_id = "echarlaix/tiny-mpt-random-remote-code"
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        onnx_config = CustomMPTOnnxConfig(
            config=config,
            task="text-generation",
            use_past=True,
            use_past_in_inputs=False,
        )
        onnx_config_with_past = CustomMPTOnnxConfig(config, task="text-generation", use_past=True)

        for legacy in (True, False):
            if legacy:
                custom_onnx_configs = {
                    "decoder_model": onnx_config,
                    "decoder_with_past_model": onnx_config_with_past,
                }
            else:
                custom_onnx_configs = {
                    "model": onnx_config_with_past,
                }

            with TemporaryDirectory() as tmpdirname:
                main_export(
                    model_id,
                    output=tmpdirname,
                    task="text-generation-with-past",
                    trust_remote_code=True,
                    custom_onnx_configs=custom_onnx_configs,
                    no_post_process=True,
                    fn_get_submodels=partial(fn_get_submodels, legacy=legacy) if fn_get_submodels else None,
                    legacy=legacy,
                    opset=14,
                )

    def test_custom_export_trust_remote_error(self):
        model_id = "mohitsha/tiny-ernie-random-remote-code"

        with self.assertRaises(ValueError) as context:
            with TemporaryDirectory() as tmpdirname:
                main_export(
                    model_id,
                    output=tmpdirname,
                    task="text-generation-with-past",
                    trust_remote_code=True,
                    no_post_process=True,
                )

        self.assertIn("custom or unsupported architecture", str(context.exception))


class OnnxExportModelTest(TestCase):
    """
    Integration tests ensuring supported models are correctly exported with export_model
    """

    def _onnx_export(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
        monolith: bool,
        device="cpu",
    ):
        library_name = TasksManager.infer_library_from_model(model_name)
        loading_kwargs = {"attn_implementation": "eager"} if model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED else {}

        if library_name == "timm":
            model_class = TasksManager.get_model_class_for_task(task, library=library_name)
            model = model_class(f"hf_hub:{model_name}", pretrained=True, exportable=True)
            TasksManager.standardize_model_attributes(model, library_name=library_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            model_class = TasksManager.get_model_class_for_task(task, model_type=config.model_type.replace("_", "-"))
            model = model_class.from_pretrained(model_name, **loading_kwargs)

        # Dynamic axes aren't supported for YOLO-like models. This means they cannot be exported to ONNX on CUDA devices.
        # See: https://github.com/ultralytics/yolov5/pull/8378
        if model.__class__.__name__.startswith("Yolos") and device != "cpu":
            return

        if model.config.model_type == "speecht5":
            model_kwargs = {"vocoder": "fxmarty/speecht5-hifigan-tiny"}
        else:
            model_kwargs = None

        if model.config.model_type == "pix2struct":
            preprocessors = maybe_load_preprocessors(model_name)
        else:
            preprocessors = None

        with TemporaryDirectory() as tmpdirname:
            onnx_export_from_model(
                model=model,
                output=Path(tmpdirname),
                monolith=monolith,
                do_validation=True,
                model_kwargs=model_kwargs,
                device=device,
                preprocessors=preprocessors,
                task=task,
            )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    @slow
    def test_pytorch_export_on_cpu(
        self,
        test_name,
        model_type,
        model_name,
        task,
        onnx_config_class_constructor,
        monolith,
    ):
        if model_type == "speecht5" and monolith:
            self.skipTest("unsupported export")

        self._onnx_export(
            test_name,
            model_type,
            model_name,
            task,
            monolith,
            device="cpu",
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    @require_torch_gpu
    @slow
    @pytest.mark.run_slow
    @pytest.mark.gpu_test
    def test_pytorch_export_on_cuda(
        self,
        test_name,
        model_type,
        model_name,
        task,
        onnx_config_class_constructor,
        monolith,
    ):
        if model_type == "speecht5" and monolith:
            self.skipTest("unsupported export")

        self._onnx_export(
            test_name,
            model_type,
            model_name,
            task,
            monolith,
            device="cuda",
        )
