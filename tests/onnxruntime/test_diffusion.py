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
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch
import torch.version
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import load_image
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from parameterized import parameterized
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin, TemporaryHubRepo

from optimum.onnxruntime import (
    ORTDiffusionPipeline,
    ORTPipelineForImage2Image,
    ORTPipelineForInpainting,
    ORTPipelineForText2Image,
)
from optimum.onnxruntime.modeling_diffusion import ORTTextEncoder, ORTUnet, ORTVae, ORTVaeDecoder, ORTVaeEncoder
from optimum.onnxruntime.utils import get_device_for_provider
from optimum.utils import is_tensorrt_available, is_transformers_version
from optimum.utils.testing_utils import (
    grid_parameters,
    remove_directory,
    require_diffusers,
    require_hf_token,
)


PROVIDERS = ["CPUExecutionProvider"]

if torch.cuda.is_available():
    if torch.version.hip is None:
        PROVIDERS.append("CUDAExecutionProvider")
    else:
        PROVIDERS.append("ROCMExecutionProvider")

    if is_tensorrt_available():
        PROVIDERS.append("TensorrtExecutionProvider")


def get_generator(framework, seed):
    if framework == "np":
        return np.random.RandomState(seed)
    elif framework == "pt":
        return torch.Generator().manual_seed(seed)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def generate_prompts(batch_size=1):
    inputs = {
        "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "np",
    }

    return inputs


IMAGE = None


def generate_images(height=128, width=128, batch_size=1, channel=3, input_type="pil"):
    if input_type == "pil":
        global IMAGE
        if IMAGE is None:
            # Load a sample image from the Hugging Face Hub
            IMAGE = load_image(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo.png"
            )
        image = IMAGE.resize((width, height))
    elif input_type == "np":
        image = np.random.rand(height, width, channel)
    elif input_type == "pt":
        image = torch.rand((channel, height, width))

    return [image] * batch_size


class ORTDiffusionPipelineTest(TestCase):
    TINY_TORCH_STABLE_DIFFUSION = "hf-internal-testing/tiny-stable-diffusion-torch"
    TINY_ONNX_STABLE_DIFFUSION = "optimum-internal-testing/tiny-stable-diffusion-onnx"

    def assert_pipeline_sanity(self, pipe: ORTDiffusionPipeline):
        self.assertIsInstance(pipe.vae, ORTVae)
        self.assertIsInstance(pipe.unet, ORTUnet)
        self.assertIsInstance(pipe.vae_encoder, ORTVaeEncoder)
        self.assertIsInstance(pipe.vae_decoder, ORTVaeDecoder)
        self.assertIsInstance(pipe.text_encoder, ORTTextEncoder)
        pipe(prompt="This is a sanity test prompt", num_inference_steps=2)

    @require_diffusers
    def test_load_diffusion_pipeline_model_from_hub(self):
        pipe = ORTDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION)
        self.assert_pipeline_sanity(pipe)

    @require_diffusers
    def test_load_diffusion_pipeline_from_path(self):
        path = snapshot_download(repo_id=self.TINY_ONNX_STABLE_DIFFUSION, allow_patterns=["*.onnx", "*.json", "*.txt"])
        pipe = ORTDiffusionPipeline.from_pretrained(path, local_files_only=True)
        self.assert_pipeline_sanity(pipe)

    @require_diffusers
    def test_load_diffusion_pipeline_from_cache(self):
        dirpath = os.path.join(HF_HUB_CACHE, "models--" + self.TINY_ONNX_STABLE_DIFFUSION.replace("/", "--"))
        if os.path.exists(dirpath):
            remove_directory(dirpath)
        with self.assertRaises(Exception):
            _ = ORTDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION, local_files_only=True)

        snapshot_download(repo_id=self.TINY_ONNX_STABLE_DIFFUSION, allow_patterns=["*.onnx", "*.json", "*.txt"])
        pipe = ORTDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION, local_files_only=True)
        self.assert_pipeline_sanity(pipe)

    @parameterized.expand(PROVIDERS)
    @require_diffusers
    def test_load_diffusion_pipeline_with_available_provider(self, provider):
        pipe = ORTDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION, provider=provider)
        self.assertEqual(pipe.device, get_device_for_provider(provider, {}))
        self.assertEqual(pipe.provider, provider)

    @require_diffusers
    def test_load_diffusion_pipeline_with_unknown_provider(self):
        with self.assertRaises(ValueError):
            ORTDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION, provider="FooExecutionProvider")

    @require_diffusers
    def test_save_diffusion_pipeline(self):
        with TemporaryDirectory() as tmpdirname:
            pipe = ORTDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION)
            pipe.save_pretrained(tmpdirname)

            folder_contents = os.listdir(tmpdirname)
            self.assertIn("model_index.json", folder_contents)
            for model_name in {"unet", "vae_encoder", "vae_decoder", "text_encoder"}:
                subfolder_contents = os.listdir(os.path.join(tmpdirname, model_name))
                self.assertIn("config.json", subfolder_contents)
                self.assertIn("model.onnx", subfolder_contents)

            for submodel_name in {"feature_extractor", "scheduler", "tokenizer"}:
                subfolder_contents = os.listdir(os.path.join(tmpdirname, submodel_name))
                if submodel_name == "scheduler":
                    self.assertIn("scheduler_config.json", subfolder_contents)
                elif submodel_name == "tokenizer":
                    self.assertIn("tokenizer_config.json", subfolder_contents)
                elif submodel_name == "feature_extractor":
                    self.assertIn("preprocessor_config.json", subfolder_contents)

            # verify reloading without export
            pipe = ORTDiffusionPipeline.from_pretrained(tmpdirname, export=False)
            self.assert_pipeline_sanity(pipe)

    @require_diffusers
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_save_diffusion_pipeline_with_external_data(self):
        with TemporaryDirectory() as tmpdirname:
            pipe = ORTDiffusionPipeline.from_pretrained(self.TINY_TORCH_STABLE_DIFFUSION, export=True)
            pipe.save_pretrained(tmpdirname)

            folder_contents = os.listdir(tmpdirname)
            self.assertIn("model_index.json", folder_contents)
            for subfoler in {"unet", "vae_encoder", "vae_decoder", "text_encoder"}:
                subfoler_contents = os.listdir(os.path.join(tmpdirname, subfoler))
                self.assertIn("model.onnx", subfoler_contents)
                self.assertIn("model.onnx_data", subfoler_contents)

            # verify reloading without export
            pipe = ORTDiffusionPipeline.from_pretrained(tmpdirname, export=False)
            self.assert_pipeline_sanity(pipe)
            remove_directory(tmpdirname)

    @require_hf_token
    @require_diffusers
    def test_push_diffusion_pipeline(self):
        # using save_pretrained(..., push_to_hub=True)
        with TemporaryHubRepo(token=os.environ.get("HF_AUTH_TOKEN", None)) as tmp_repo:
            pipe = ORTDiffusionPipeline.from_pretrained(self.TINY_TORCH_STABLE_DIFFUSION, export=True)
            pipe.save_pretrained(tmp_repo.repo_id, token=os.environ.get("HF_AUTH_TOKEN", None), push_to_hub=True)
            pipe = ORTDiffusionPipeline.from_pretrained(
                tmp_repo.repo_id, token=os.environ.get("HF_AUTH_TOKEN", None), export=False
            )
            self.assert_pipeline_sanity(pipe)

        # using push_to_hub(...)
        with TemporaryHubRepo(token=os.environ.get("HF_AUTH_TOKEN", None)) as tmp_repo:
            pipe.push_to_hub(tmp_repo.repo_id, token=os.environ.get("HF_AUTH_TOKEN", None))
            pipe = ORTDiffusionPipeline.from_pretrained(
                tmp_repo.repo_id, token=os.environ.get("HF_AUTH_TOKEN", None), export=False
            )
            self.assert_pipeline_sanity(pipe)


class ORTPipelineForText2ImageTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
        "stable-diffusion-xl",
        "latent-consistency",
    ]
    if is_transformers_version(">=", "4.45"):
        SUPPORTED_ARCHITECTURES += ["stable-diffusion-3", "flux"]

    NEGATIVE_PROMPT_SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
        "stable-diffusion-xl",
        "latent-consistency",
    ]

    if is_transformers_version(">=", "4.45"):
        NEGATIVE_PROMPT_SUPPORTED_ARCHITECTURES += ["stable-diffusion-3"]

    CALLBACK_SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
        "stable-diffusion-xl",
        "latent-consistency",
    ]
    if is_transformers_version(">=", "4.45"):
        CALLBACK_SUPPORTED_ARCHITECTURES += ["flux"]

    ORTMODEL_CLASS = ORTPipelineForText2Image
    AUTOMODEL_CLASS = AutoPipelineForText2Image

    TASK = "text-to-image"

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = generate_prompts(batch_size=batch_size)

        inputs["height"], inputs["width"] = height, width

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"])

        self.assertIn(
            f"does not appear to have a file named {self.ORTMODEL_CLASS.config_name}", str(context.exception)
        )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_class_dispatch(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        self.assertEqual(ort_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": PROVIDERS}))
    @require_diffusers
    def test_ort_pipeline(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("Testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 32, 1
        device = get_device_for_provider(provider, {})
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        self.assertEqual(pipeline.device, device)

        outputs = pipeline(**inputs).images
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for batch_size in [1, 3]:
            for height in [16, 32]:
                for width in [16, 32]:
                    for num_images_per_prompt in [1, 3]:
                        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
                        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
                        self.assertEqual(outputs.shape, (batch_size * num_images_per_prompt, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 128, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type

            ort_images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_images = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ort_images, diffusers_images, atol=1e-4, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_io_binding(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusion_model = ort_pipeline.unet or ort_pipeline.transformer

        for output_type in ["latent", "pt"]:
            inputs["output_type"] = output_type

            # makes sure io binding is not used
            ort_pipeline.use_io_binding = False
            images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            self.assertEqual(len(diffusion_model._io_binding.get_outputs()), 0)

            # makes sure io binding is effectively used
            ort_pipeline.use_io_binding = True
            io_images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            self.assertGreaterEqual(len(diffusion_model._io_binding.get_outputs()), 1)

            # makes sure the outputs are the same
            np.testing.assert_allclose(images, io_images, atol=1e-4, rtol=1e-2)
            # clears the io binding outputs
            diffusion_model._io_binding.clear_binding_outputs()

    @parameterized.expand(CALLBACK_SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_callback(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        class Callback:
            def __init__(self):
                self.has_been_called = False
                self.number_of_steps = 0

            def __call__(self, *args, **kwargs) -> None:
                self.has_been_called = True
                self.number_of_steps += 1
                return kwargs

        ort_callback = Callback()
        auto_callback = Callback()

        ort_pipe = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipe = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        ort_pipe(**inputs, callback_on_step_end=ort_callback)
        auto_pipe(**inputs, callback_on_step_end=auto_callback)

        self.assertTrue(ort_callback.has_been_called)
        self.assertTrue(auto_callback.has_been_called)
        self.assertEqual(auto_callback.number_of_steps, ort_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusion_model = pipeline.unet or pipeline.transformer
        height, width, batch_size = 64, 64, 1

        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        for output_type in ["pil", "np", "pt", "latent"]:
            inputs["output_type"] = output_type
            outputs = pipeline(**inputs).images

            if output_type == "pil":
                self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
            elif output_type == "np":
                self.assertEqual(outputs.shape, (batch_size, height, width, 3))
            elif output_type == "pt":
                self.assertEqual(outputs.shape, (batch_size, 3, height, width))
            else:
                if model_arch == "flux":
                    expected_height = height // (pipeline.vae_scale_factor * 2)
                    expected_width = width // (pipeline.vae_scale_factor * 2)
                    channels = pipeline.transformer.config.in_channels
                    expected_shape = (batch_size, expected_height * expected_width, channels)
                else:
                    expected_height = height // pipeline.vae_scale_factor
                    expected_width = width // pipeline.vae_scale_factor
                    out_channels = diffusion_model.config.out_channels
                    expected_shape = (batch_size, out_channels, expected_height, expected_width)

                self.assertEqual(outputs.shape, expected_shape)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for generator_framework in ["np", "pt"]:
            ort_outputs_1 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_2 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_3 = pipeline(**inputs, generator=get_generator(generator_framework, SEED + 1))

            self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))
            np.testing.assert_allclose(ort_outputs_1.images[0], ort_outputs_2.images[0], atol=1e-4, rtol=1e-2)

    @parameterized.expand(NEGATIVE_PROMPT_SUPPORTED_ARCHITECTURES)
    def test_negative_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        inputs["negative_prompt"] = ["This is a negative prompt"] * batch_size

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        ort_images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
        diffusers_images = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

        np.testing.assert_allclose(ort_images, diffusers_images, atol=1e-4, rtol=1e-2)

    @parameterized.expand(["stable-diffusion", "latent-consistency"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "katuni4ka/tiny-random-stable-diffusion-with-safety-checker", subfolder="safety_checker"
        )

        pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch], safety_checker=safety_checker
        )

        self.assertIsInstance(pipeline.safety_checker, StableDiffusionSafetyChecker)
        self.assertIsInstance(ort_pipeline.safety_checker, StableDiffusionSafetyChecker)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_output = ort_pipeline(**inputs, generator=get_generator("pt", SEED))
        diffusers_output = pipeline(**inputs, generator=get_generator("pt", SEED))

        ort_nsfw_content_detected = ort_output.nsfw_content_detected
        diffusers_nsfw_content_detected = diffusers_output.nsfw_content_detected

        self.assertTrue(ort_nsfw_content_detected is not None)
        self.assertTrue(diffusers_nsfw_content_detected is not None)
        self.assertEqual(ort_nsfw_content_detected, diffusers_nsfw_content_detected)

        ort_images = ort_output.images
        diffusers_images = diffusers_output.images
        np.testing.assert_allclose(ort_images, diffusers_images, atol=1e-4, rtol=1e-2)


class ORTPipelineForImage2ImageTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
        "stable-diffusion-xl",
        "latent-consistency",
    ]
    if is_transformers_version(">=", "4.45"):
        SUPPORTED_ARCHITECTURES += ["stable-diffusion-3"]

    CALLBACK_SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
        "stable-diffusion-xl",
        "latent-consistency",
    ]

    AUTOMODEL_CLASS = AutoPipelineForImage2Image
    ORTMODEL_CLASS = ORTPipelineForImage2Image

    TASK = "image-to-image"

    def generate_inputs(self, height=128, width=128, batch_size=1, channel=3, input_type="pil"):
        inputs = generate_prompts(batch_size=batch_size)

        inputs["image"] = generate_images(
            height=height, width=width, batch_size=batch_size, channel=channel, input_type=input_type
        )

        inputs["height"], inputs["width"] = height, width
        inputs["strength"] = 0.75

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"])

        self.assertIn(
            f"does not appear to have a file named {self.ORTMODEL_CLASS.config_name}", str(context.exception)
        )

    @parameterized.expand(list(SUPPORTED_ARCHITECTURES))
    @require_diffusers
    def test_class_dispatch(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        self.assertEqual(ort_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": PROVIDERS}))
    @require_diffusers
    def test_ort_pipeline(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("Testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 32, 1
        device = get_device_for_provider(provider, {})
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        self.assertEqual(pipeline.device, device)

        outputs = pipeline(**inputs).images
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for batch_size in [1, 3]:
            for height in [16, 32]:
                for width in [16, 32]:
                    for num_images_per_prompt in [1, 3]:
                        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
                        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
                        self.assertEqual(outputs.shape, (batch_size * num_images_per_prompt, height, width, 3))

    @parameterized.expand(CALLBACK_SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_callback(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        inputs["num_inference_steps"] = 3

        class Callback:
            def __init__(self):
                self.has_been_called = False
                self.number_of_steps = 0

            def __call__(self, *args, **kwargs) -> None:
                self.has_been_called = True
                self.number_of_steps += 1
                return kwargs

        ort_pipe = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipe = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        ort_callback = Callback()
        auto_callback = Callback()

        ort_pipe(**inputs, callback_on_step_end=ort_callback)
        auto_pipe(**inputs, callback_on_step_end=auto_callback)

        self.assertTrue(ort_callback.has_been_called)
        self.assertEqual(ort_callback.number_of_steps, auto_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusion_model = pipeline.unet or pipeline.transformer
        height, width, batch_size = 64, 64, 1

        for input_type in ["pil", "np", "pt"]:
            inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, input_type=input_type)

            for output_type in ["pil", "np", "pt", "latent"]:
                inputs["output_type"] = output_type
                outputs = pipeline(**inputs).images

                if output_type == "pil":
                    self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
                elif output_type == "np":
                    self.assertEqual(outputs.shape, (batch_size, height, width, 3))
                elif output_type == "pt":
                    self.assertEqual(outputs.shape, (batch_size, 3, height, width))
                else:
                    self.assertEqual(
                        outputs.shape,
                        (
                            batch_size,
                            diffusion_model.config.out_channels,
                            height // pipeline.vae_scale_factor,
                            width // pipeline.vae_scale_factor,
                        ),
                    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 128, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type

            ort_images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_images = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ort_images, diffusers_images, atol=1e-4, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_io_binding(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusion_model = ort_pipeline.unet or ort_pipeline.transformer

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type

            # makes sure io binding is not used
            ort_pipeline.use_io_binding = False
            images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            self.assertEqual(len(diffusion_model._io_binding.get_outputs()), 0)

            # makes sure io binding is effectively used
            ort_pipeline.use_io_binding = True
            io_images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            self.assertGreaterEqual(len(diffusion_model._io_binding.get_outputs()), 1)

            # makes sure the outputs are the same
            np.testing.assert_allclose(images, io_images, atol=1e-4, rtol=1e-2)
            # clears the io binding outputs
            diffusion_model._io_binding.clear_binding_outputs()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for generator_framework in ["np", "pt"]:
            ort_outputs_1 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_2 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_3 = pipeline(**inputs, generator=get_generator(generator_framework, SEED + 1))

            self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))
            np.testing.assert_allclose(ort_outputs_1.images[0], ort_outputs_2.images[0], atol=1e-4, rtol=1e-2)

    @parameterized.expand(["stable-diffusion", "latent-consistency"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "katuni4ka/tiny-random-stable-diffusion-with-safety-checker", subfolder="safety_checker"
        )

        pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch], safety_checker=safety_checker
        )

        self.assertIsInstance(pipeline.safety_checker, StableDiffusionSafetyChecker)
        self.assertIsInstance(ort_pipeline.safety_checker, StableDiffusionSafetyChecker)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_output = ort_pipeline(**inputs, generator=get_generator("pt", SEED))
        diffusers_output = pipeline(**inputs, generator=get_generator("pt", SEED))

        ort_nsfw_content_detected = ort_output.nsfw_content_detected
        diffusers_nsfw_content_detected = diffusers_output.nsfw_content_detected

        self.assertTrue(ort_nsfw_content_detected is not None)
        self.assertTrue(diffusers_nsfw_content_detected is not None)
        self.assertEqual(ort_nsfw_content_detected, diffusers_nsfw_content_detected)

        ort_images = ort_output.images
        diffusers_images = diffusers_output.images

        np.testing.assert_allclose(ort_images, diffusers_images, atol=1e-4, rtol=1e-2)


class ORTPipelineForInpaintingTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
        "stable-diffusion-xl",
    ]
    if is_transformers_version(">=", "4.45"):
        SUPPORTED_ARCHITECTURES += ["stable-diffusion-3"]

    CALLBACK_SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
        "stable-diffusion-xl",
    ]

    AUTOMODEL_CLASS = AutoPipelineForInpainting
    ORTMODEL_CLASS = ORTPipelineForInpainting

    TASK = "inpainting"

    def generate_inputs(self, height=128, width=128, batch_size=1, channel=3, input_type="pil"):
        inputs = generate_prompts(batch_size=batch_size)

        inputs["image"] = generate_images(
            height=height, width=width, batch_size=batch_size, channel=channel, input_type=input_type
        )
        inputs["mask_image"] = generate_images(
            height=height, width=width, batch_size=batch_size, channel=1, input_type=input_type
        )

        inputs["height"], inputs["width"] = height, width
        inputs["strength"] = 0.75

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"])

        self.assertIn(
            f"does not appear to have a file named {self.ORTMODEL_CLASS.config_name}", str(context.exception)
        )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_class_dispatch(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        self.assertEqual(ort_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": PROVIDERS}))
    @require_diffusers
    def test_ort_pipeline(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("Testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 32, 1
        device = get_device_for_provider(provider, {})
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        self.assertEqual(pipeline.device, device)

        outputs = pipeline(**inputs).images
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for batch_size in [1, 3]:
            for height in [16, 32]:
                for width in [16, 32]:
                    for num_images_per_prompt in [1, 3]:
                        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
                        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
                        self.assertEqual(outputs.shape, (batch_size * num_images_per_prompt, height, width, 3))

    @parameterized.expand(CALLBACK_SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_callback(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        inputs["num_inference_steps"] = 3

        class Callback:
            def __init__(self):
                self.has_been_called = False
                self.number_of_steps = 0

            def __call__(self, *args, **kwargs) -> None:
                self.has_been_called = True
                self.number_of_steps += 1
                return kwargs

        ort_pipe = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipe = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        ort_callback = Callback()
        auto_callback = Callback()

        ort_pipe(**inputs, callback_on_step_end=ort_callback)
        auto_pipe(**inputs, callback_on_step_end=auto_callback)

        self.assertTrue(ort_callback.has_been_called)
        self.assertEqual(ort_callback.number_of_steps, auto_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusion_model = pipeline.unet or pipeline.transformer
        height, width, batch_size = 64, 64, 1

        for input_type in ["pil", "np", "pt"]:
            inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, input_type=input_type)

            for output_type in ["pil", "np", "pt", "latent"]:
                inputs["output_type"] = output_type
                outputs = pipeline(**inputs).images

                if output_type == "pil":
                    self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
                elif output_type == "np":
                    self.assertEqual(outputs.shape, (batch_size, height, width, 3))
                elif output_type == "pt":
                    self.assertEqual(outputs.shape, (batch_size, 3, height, width))
                else:
                    self.assertEqual(
                        outputs.shape,
                        (
                            batch_size,
                            diffusion_model.config.out_channels,
                            height // pipeline.vae_scale_factor,
                            width // pipeline.vae_scale_factor,
                        ),
                    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type

            ort_images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_images = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ort_images, diffusers_images, atol=1e-4, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_io_binding(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusion_model = ort_pipeline.unet or ort_pipeline.transformer

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type

            # makes sure io binding is not used
            ort_pipeline.use_io_binding = False
            images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            self.assertEqual(len(diffusion_model._io_binding.get_outputs()), 0)

            # makes sure io binding is effectively used
            ort_pipeline.use_io_binding = True
            io_images = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            self.assertGreaterEqual(len(diffusion_model._io_binding.get_outputs()), 1)

            # makes sure the outputs are the same
            np.testing.assert_allclose(images, io_images, atol=1e-4, rtol=1e-2)
            # clears the io binding outputs
            diffusion_model._io_binding.clear_binding_outputs()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for generator_framework in ["np", "pt"]:
            ort_outputs_1 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_2 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_3 = pipeline(**inputs, generator=get_generator(generator_framework, SEED + 1))

            self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))
            np.testing.assert_allclose(ort_outputs_1.images[0], ort_outputs_2.images[0], atol=1e-4, rtol=1e-2)

    @parameterized.expand(["stable-diffusion"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "katuni4ka/tiny-random-stable-diffusion-with-safety-checker", subfolder="safety_checker"
        )

        pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch], safety_checker=safety_checker
        )

        self.assertIsInstance(pipeline.safety_checker, StableDiffusionSafetyChecker)
        self.assertIsInstance(ort_pipeline.safety_checker, StableDiffusionSafetyChecker)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_output = ort_pipeline(**inputs, generator=get_generator("pt", SEED))
        diffusers_output = pipeline(**inputs, generator=get_generator("pt", SEED))

        ort_nsfw_content_detected = ort_output.nsfw_content_detected
        diffusers_nsfw_content_detected = diffusers_output.nsfw_content_detected

        self.assertTrue(ort_nsfw_content_detected is not None)
        self.assertTrue(diffusers_nsfw_content_detected is not None)
        self.assertEqual(ort_nsfw_content_detected, diffusers_nsfw_content_detected)

        ort_images = ort_output.images
        diffusers_images = diffusers_output.images
        np.testing.assert_allclose(ort_images, diffusers_images, atol=1e-4, rtol=1e-2)
