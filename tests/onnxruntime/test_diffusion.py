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

import numpy as np
import pytest
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import load_image
from parameterized import parameterized
from transformers.testing_utils import require_torch_gpu
from utils_onnxruntime_tests import MODEL_NAMES, SEED, ORTModelTestMixin

from optimum.onnxruntime import (
    ORTDiffusionPipeline,
    ORTPipelineForImage2Image,
    ORTPipelineForInpainting,
    ORTPipelineForText2Image,
)
from optimum.utils.testing_utils import grid_parameters, require_diffusers


def get_generator(framework, seed):
    if framework == "np":
        return np.random.RandomState(seed)
    elif framework == "pt":
        return torch.Generator().manual_seed(seed)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def _generate_prompts(batch_size=1):
    inputs = {
        "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "np",
    }
    return inputs


def _generate_images(height=128, width=128, batch_size=1, channel=3, input_type="pil"):
    if input_type == "pil":
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        ).resize((width, height))
    elif input_type == "np":
        image = np.random.rand(height, width, channel)
    elif input_type == "pt":
        image = torch.rand((channel, height, width))

    return [image] * batch_size


class ORTPipelineForText2ImageTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl", "latent-consistency"]

    ORTMODEL_CLASS = ORTPipelineForText2Image
    AUTOMODEL_CLASS = AutoPipelineForText2Image

    TASK = "text-to-image"

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = _generate_prompts(batch_size=batch_size)

        inputs["height"] = height
        inputs["width"] = width

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn(
            f"does not appear to have a file named {self.ORTMODEL_CLASS.config_name}", str(context.exception)
        )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_ort_pipeline_class_dispatch(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        auto_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertEqual(ort_pipeline.auto_model_class, auto_pipeline.__class__)

        auto_pipeline = DiffusionPipeline.from_pretrained(MODEL_NAMES[model_arch])
        ort_pipeline = ORTDiffusionPipeline.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertEqual(ort_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for batch_size in [1, 3]:
            for height in [64, 128]:
                for width in [64, 128]:
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

            ort_output = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ort_output, diffusers_output, atol=1e-4, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
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

        ort_callback = Callback()
        auto_callback = Callback()

        ort_pipe = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipe = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        # callback_steps=1 to trigger callback every step
        ort_pipe(**inputs, callback=ort_callback, callback_steps=1)
        auto_pipe(**inputs, callback=auto_callback, callback_steps=1)

        self.assertTrue(ort_callback.has_been_called)
        self.assertTrue(auto_callback.has_been_called)
        self.assertEqual(auto_callback.number_of_steps, ort_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        height, width, batch_size = 128, 64, 1
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
                self.assertEqual(
                    outputs.shape,
                    (batch_size, 4, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
                )

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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_negative_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        negative_prompt = ["This is a negative prompt"]
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        images_1 = pipeline(**inputs, negative_prompt=negative_prompt, generator=get_generator("pt", SEED)).images
        prompt = inputs.pop("prompt")

        if model_arch == "stable-diffusion-xl":
            (
                inputs["prompt_embeds"],
                inputs["negative_prompt_embeds"],
                inputs["pooled_prompt_embeds"],
                inputs["negative_pooled_prompt_embeds"],
            ) = pipeline.encode_prompt(
                prompt=prompt,
                num_images_per_prompt=1,
                device=torch.device("cpu"),
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        else:
            inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = pipeline.encode_prompt(
                prompt=prompt,
                num_images_per_prompt=1,
                device=torch.device("cpu"),
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        images_2 = pipeline(**inputs, generator=get_generator("pt", SEED)).images

        np.testing.assert_allclose(images_1, images_2, atol=1e-4, rtol=1e-2)

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "provider": ["CUDAExecutionProvider", "ROCMExecutionProvider", "TensorrtExecutionProvider"],
            }
        )
    )
    @pytest.mark.rocm_ep_test
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    @require_torch_gpu
    @require_diffusers
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)

        outputs = pipeline(**inputs).images

        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(["stable-diffusion", "latent-consistency"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")

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
    SUPPORTED_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl", "latent-consistency"]

    AUTOMODEL_CLASS = AutoPipelineForImage2Image
    ORTMODEL_CLASS = ORTPipelineForImage2Image

    TASK = "image-to-image"

    def generate_inputs(self, height=128, width=128, batch_size=1, channel=3, input_type="pil"):
        inputs = _generate_prompts(batch_size=batch_size)

        inputs["image"] = _generate_images(
            height=height, width=width, batch_size=batch_size, channel=channel, input_type=input_type
        )

        inputs["strength"] = 0.75

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn(
            f"does not appear to have a file named {self.ORTMODEL_CLASS.config_name}", str(context.exception)
        )

    @parameterized.expand(list(SUPPORTED_ARCHITECTURES))
    @require_diffusers
    def test_ort_pipeline_class_dispatch(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        auto_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertEqual(ort_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for batch_size in [1, 3]:
            for height in [64, 128]:
                for width in [64, 128]:
                    for num_images_per_prompt in [1, 3]:
                        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
                        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
                        self.assertEqual(outputs.shape, (batch_size * num_images_per_prompt, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
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

        ort_pipe = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipe = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        ort_callback = Callback()
        auto_callback = Callback()
        # callback_steps=1 to trigger callback every step
        ort_pipe(**inputs, callback=ort_callback, callback_steps=1)
        auto_pipe(**inputs, callback=auto_callback, callback_steps=1)

        self.assertTrue(ort_callback.has_been_called)
        self.assertEqual(ort_callback.number_of_steps, auto_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        height, width, batch_size = 128, 64, 1

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
                        (batch_size, 4, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
                    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 128, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type

            ort_output = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ort_output, diffusers_output, atol=1e-4, rtol=1e-2)

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

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "provider": ["CUDAExecutionProvider", "ROCMExecutionProvider", "TensorrtExecutionProvider"],
            }
        )
    )
    @pytest.mark.rocm_ep_test
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    @require_torch_gpu
    @require_diffusers
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        self.assertEqual(pipeline.device.type, "cuda")

        outputs = pipeline(**inputs).images
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(["stable-diffusion", "latent-consistency"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")

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
    SUPPORTED_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl"]

    AUTOMODEL_CLASS = AutoPipelineForInpainting
    ORTMODEL_CLASS = ORTPipelineForInpainting

    TASK = "inpainting"

    def generate_inputs(self, height=128, width=128, batch_size=1, channel=3, input_type="pil"):
        inputs = _generate_prompts(batch_size=batch_size)

        inputs["image"] = _generate_images(
            height=height, width=width, batch_size=batch_size, channel=channel, input_type=input_type
        )
        inputs["mask_image"] = _generate_images(
            height=height, width=width, batch_size=batch_size, channel=1, input_type=input_type
        )

        inputs["strength"] = 0.75
        inputs["height"] = height
        inputs["width"] = width

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn(
            f"does not appear to have a file named {self.ORTMODEL_CLASS.config_name}", str(context.exception)
        )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_ort_pipeline_class_dispatch(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        auto_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertEqual(ort_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for batch_size in [1, 3]:
            for height in [64, 128]:
                for width in [64, 128]:
                    for num_images_per_prompt in [1, 3]:
                        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
                        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
                        self.assertEqual(outputs.shape, (batch_size * num_images_per_prompt, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
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

        ort_pipe = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        auto_pipe = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        ort_callback = Callback()
        auto_callback = Callback()
        # callback_steps=1 to trigger callback every step
        ort_pipe(**inputs, callback=ort_callback, callback_steps=1)
        auto_pipe(**inputs, callback=auto_callback, callback_steps=1)

        self.assertTrue(ort_callback.has_been_called)
        self.assertEqual(ort_callback.number_of_steps, auto_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        height, width, batch_size = 128, 64, 1

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
                        (batch_size, 4, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
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

            ort_output = ort_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ort_output, diffusers_output, atol=1e-4, rtol=1e-2)

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

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "provider": ["CUDAExecutionProvider", "ROCMExecutionProvider", "TensorrtExecutionProvider"],
            }
        )
    )
    @pytest.mark.rocm_ep_test
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    @require_torch_gpu
    @require_diffusers
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        self.assertEqual(pipeline.device, "cuda")

        outputs = pipeline(**inputs).images
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(["stable-diffusion"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")

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
