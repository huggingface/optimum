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
import unittest

import numpy as np
import PIL
import pytest
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)
from diffusers.utils import load_image
from parameterized import parameterized
from transformers.testing_utils import require_torch_gpu
from utils_onnxruntime_tests import MODEL_NAMES, SEED, ORTModelTestMixin

from optimum.onnxruntime import (
    ORTLatentConsistencyModelPipeline,
    ORTPipelineForImage2Image,
    ORTPipelineForInpainting,
    ORTPipelineForText2Image,
    ORTStableDiffusionImg2ImgPipeline,
    ORTStableDiffusionInpaintPipeline,
    ORTStableDiffusionPipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
    ORTStableDiffusionXLPipeline,
)
from optimum.pipelines.diffusers.pipeline_utils import VaeImageProcessor
from optimum.utils.testing_utils import grid_parameters, require_diffusers, require_ort_rocm


def get_generator(generator_framework, seed):
    if generator_framework == "np":
        return np.random.RandomState(seed)
    elif generator_framework == "pt":
        return torch.Generator().manual_seed(seed)
    else:
        raise ValueError(f"Unknown generator_framework: {generator_framework}")


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


def to_np(image):
    if isinstance(image[0], PIL.Image.Image):
        return np.stack([np.array(i) for i in image], axis=0)
    elif isinstance(image, torch.Tensor):
        return image.cpu().numpy().transpose(0, 2, 3, 1)
    return image


class ORTPipelineForText2ImageTest(ORTModelTestMixin):
    ARCHITECTURE_TO_ORTMODEL_CLASS = {
        "lcm": ORTLatentConsistencyModelPipeline,
        "stable-diffusion": ORTStableDiffusionPipeline,
        "stable-diffusion-xl": ORTStableDiffusionXLPipeline,
    }

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

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_ort_pipeline_class_dispatch(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        self.assertIsInstance(pipeline, self.ARCHITECTURE_TO_ORTMODEL_CLASS[model_arch])

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        self.assertEqual(pipeline.vae_scale_factor, 2)
        self.assertEqual(pipeline.vae_decoder.config["latent_channels"], 4)
        self.assertEqual(pipeline.unet.config["in_channels"], 4)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        for num_images in [1, 3]:
            outputs = pipeline(**inputs, num_images_per_prompt=num_images).images
            self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 128, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        if model_arch == "lcm":
            # LCM doesn't support deterministic outputs beyond the first inference step
            # TODO: Investigate why this is the case
            inputs["num_inference_steps"] = 1

        for output_type in ["latent", "np"]:
            inputs["output_type"] = output_type

            ort_output = ort_pipeline(**inputs, generator=torch.Generator().manual_seed(SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=torch.Generator().manual_seed(SEED)).images

            self.assertTrue(
                np.allclose(ort_output, diffusers_output, atol=1e-4),
                np.testing.assert_allclose(ort_output, diffusers_output, atol=1e-4),
            )
            self.assertEqual(ort_pipeline.device, diffusers_pipeline.device)

    @parameterized.expand(
        grid_parameters(
            {"model_arch": list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()), "provider": ["CUDAExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @require_diffusers
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        outputs = pipeline(**inputs).images
        # Verify model devices
        self.assertEqual(pipeline.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()), "provider": ["ROCMExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    @require_diffusers
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        height, width, batch_size = 64, 32, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        outputs = pipeline(**inputs).images
        # Verify model devices
        self.assertEqual(pipeline.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
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

            def __call__(self, step: int, timestep: int, latents: np.ndarray) -> None:
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

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        height, width, batch_size = 128, 64, 1
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        for output_type in ["np", "pil", "latent"]:
            inputs["output_type"] = output_type
            outputs = pipeline(**inputs).images
            if output_type == "pil":
                self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
            elif output_type == "np":
                self.assertEqual(outputs.shape, (batch_size, height, width, 3))
            else:
                self.assertEqual(
                    outputs.shape,
                    (batch_size, 4, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
                )

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        if model_arch in ["lcm"]:
            pytest.skip("Latent Consistency Model (LCM) doesn't support deterministic outputs")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for generator_framework in ["np", "pt"]:
            ort_outputs_1 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_2 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_3 = pipeline(**inputs, generator=get_generator(generator_framework, SEED + 1))

            self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
            self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    def test_negative_prompt(self, model_arch: str):
        if model_arch in ["lcm"]:
            pytest.skip("LCM (Latent Consistency Model) does not support negative prompts")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        negative_prompt = ["This is a negative prompt"]
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        image_slice_1 = pipeline(
            **inputs, negative_prompt=negative_prompt, generator=np.random.RandomState(SEED)
        ).images[0, -3:, -3:, -1]
        prompt = inputs.pop("prompt")

        if model_arch == "stable-diffusion-xl":
            (
                inputs["prompt_embeds"],
                inputs["negative_prompt_embeds"],
                inputs["pooled_prompt_embeds"],
                inputs["negative_pooled_prompt_embeds"],
            ) = pipeline._encode_prompt(prompt, 1, False, negative_prompt)
        else:
            text_ids = pipeline.tokenizer(
                prompt,
                max_length=pipeline.tokenizer.model_max_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            ).input_ids
            negative_text_ids = pipeline.tokenizer(
                negative_prompt,
                max_length=pipeline.tokenizer.model_max_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            ).input_ids
            inputs["prompt_embeds"] = pipeline.text_encoder(text_ids)[0]
            inputs["negative_prompt_embeds"] = pipeline.text_encoder(negative_text_ids)[0]

        image_slice_2 = pipeline(**inputs, generator=np.random.RandomState(SEED)).images[0, -3:, -3:, -1]

        self.assertTrue(np.allclose(image_slice_1, image_slice_2, rtol=1e-1))


class ORTPipelineForImage2ImageTest(ORTModelTestMixin):
    ARCHITECTURE_TO_ORTMODEL_CLASS = {
        "stable-diffusion": ORTStableDiffusionImg2ImgPipeline,
        "stable-diffusion-xl": ORTStableDiffusionXLImg2ImgPipeline,
    }
    AUTOMODEL_CLASS = AutoPipelineForImage2Image
    ORTMODEL_CLASS = ORTPipelineForImage2Image

    TASK = "image-to-image"

    def generate_inputs(self, height=128, width=128, batch_size=1, channel=3, input_type="np"):
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

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        self.assertEqual(pipeline.vae_scale_factor, 2)
        self.assertEqual(pipeline.vae_decoder.config["latent_channels"], 4)
        self.assertEqual(pipeline.unet.config["in_channels"], 4)

        batch_size, height = 1, 32
        for width in [64, 32]:
            inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
            for num_images in [1, 3]:
                outputs = pipeline(**inputs, num_images_per_prompt=num_images).images
                self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()), "provider": ["CUDAExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @require_diffusers
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        outputs = pipeline(**inputs).images
        # Verify model devices
        self.assertEqual(pipeline.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()), "provider": ["ROCMExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    @require_diffusers
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        outputs = pipeline(**inputs).images
        # Verify model devices
        self.assertEqual(pipeline.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_callback(self, model_arch: str):
        if model_arch in ["stable-diffusion"]:
            pytest.skip(
                "Stable Diffusion For Img2Img doesn't behave as expected with callbacks (doesn't call it every step with callback_steps=1)"
            )

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        inputs["num_inference_steps"] = 3

        class Callback:
            def __init__(self):
                self.has_been_called = False
                self.number_of_steps = 0

            def __call__(self, step: int, timestep: int, latents: np.ndarray) -> None:
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

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        height, width, batch_size = 32, 64, 1

        for input_type in ["np", "pil", "pt"]:
            inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, input_type=input_type)

            for output_type in ["np", "pil", "latent"]:
                inputs["output_type"] = output_type
                outputs = pipeline(**inputs).images
                if output_type == "pil":
                    self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
                elif output_type == "np":
                    self.assertEqual(outputs.shape, (batch_size, height, width, 3))
                else:
                    self.assertEqual(
                        outputs.shape,
                        (batch_size, 4, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
                    )

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        pytest.skip("Img2Img models do not support support output reproducibility for some reason")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 128, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        ort_output = ort_pipeline(**inputs, generator=torch.Generator().manual_seed(SEED)).images

        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        diffusers_output = diffusers_pipeline(**inputs, generator=torch.Generator().manual_seed(SEED)).images

        self.assertTrue(np.allclose(ort_output, diffusers_output, rtol=1e-2))

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        pytest.skip("Img2Img models do not support support output reproducibility for some reason")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        for generator_framework in ["np", "pt"]:
            ort_outputs_1 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_2 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ort_outputs_3 = pipeline(**inputs, generator=get_generator(generator_framework, SEED + 1))

            self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
            self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))


class ORTPipelineForInpaintingTest(ORTModelTestMixin):
    ARCHITECTURE_TO_ORTMODEL_CLASS = {
        "stable-diffusion": ORTStableDiffusionInpaintPipeline,
    }

    AUTOMODEL_CLASS = AutoPipelineForInpainting
    ORTMODEL_CLASS = ORTPipelineForInpainting

    TASK = "inpainting"

    def generate_inputs(self, height=128, width=128, batch_size=1, channel=3, input_type="pil"):
        assert batch_size == 1, "Inpainting models only support batch_size=1"
        assert input_type == "pil", "Inpainting models only support input_type='pil'"

        inputs = _generate_prompts(batch_size=batch_size)

        inputs["image"] = _generate_images(
            height=height, width=width, batch_size=1, channel=channel, input_type="pil"
        )[0]
        inputs["mask_image"] = _generate_images(
            height=height, width=width, batch_size=1, channel=channel, input_type="pil"
        )[0]

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

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        self.assertEqual(pipeline.vae_scale_factor, 2)
        self.assertEqual(pipeline.vae_decoder.config["latent_channels"], 4)
        self.assertEqual(pipeline.unet.config["in_channels"], 4)

        batch_size, height = 1, 32
        for width in [64, 32]:
            inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
            for num_images in [1, 3]:
                outputs = pipeline(**inputs, num_images_per_prompt=num_images).images
                self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()), "provider": ["CUDAExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @require_diffusers
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        outputs = pipeline(**inputs).images
        # Verify model devices
        self.assertEqual(pipeline.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()), "provider": ["ROCMExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    @require_diffusers
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        outputs = pipeline(**inputs).images
        # Verify model devices
        self.assertEqual(pipeline.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (batch_size, height, width, 3))

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
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

            def __call__(self, step: int, timestep: int, latents: np.ndarray) -> None:
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

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        height, width, batch_size = 32, 64, 1

        for input_type in ["pil"]:
            inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, input_type=input_type)

            for output_type in ["np", "pil", "latent"]:
                inputs["output_type"] = output_type
                outputs = pipeline(**inputs).images
                if output_type == "pil":
                    self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
                elif output_type == "np":
                    self.assertEqual(outputs.shape, (batch_size, height, width, 3))
                else:
                    self.assertEqual(
                        outputs.shape,
                        (batch_size, 4, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
                    )

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        if model_arch in ["stable-diffusion"]:
            pytest.skip(
                "Stable Diffusion For Inpainting fails, it was used to be compared to StableDiffusionPipeline for some reason which is the text-to-image variant"
            )

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        latents_shape = (
            batch_size,
            ort_pipeline.vae_decoder.config["latent_channels"],
            height // ort_pipeline.vae_scale_factor,
            width // ort_pipeline.vae_scale_factor,
        )

        np_latents = np.random.rand(*latents_shape).astype(np.float32)
        torch_latents = torch.from_numpy(np_latents)

        ort_output = ort_pipeline(**inputs, latents=np_latents).images
        diffusers_output = diffusers_pipeline(**inputs, latents=torch_latents).images

        self.assertTrue(
            np.allclose(ort_output, diffusers_output, atol=1e-4),
            np.testing.assert_allclose(ort_output, diffusers_output, atol=1e-4),
        )

    @parameterized.expand(list(ARCHITECTURE_TO_ORTMODEL_CLASS.keys()))
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

            self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
            self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))


class ImageProcessorTest(unittest.TestCase):
    def test_vae_image_processor_pt(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
        input_pt = torch.stack(_generate_images(height=8, width=8, batch_size=1, input_type="pt"))
        input_np = to_np(input_pt)

        for output_type in ["np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_pt), output_type=output_type)
            out_np = to_np(out)
            in_np = (input_np * 255).round() if output_type == "pil" else input_np
            self.assertTrue(np.allclose(in_np, out_np, atol=1e-6))

    def test_vae_image_processor_np(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
        input_np = np.stack(_generate_images(height=8, width=8, input_type="np"))
        for output_type in ["np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_np), output_type=output_type)
            out_np = to_np(out)
            in_np = (input_np * 255).round() if output_type == "pil" else input_np
            self.assertTrue(np.allclose(in_np, out_np, atol=1e-6))

    def test_vae_image_processor_pil(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
        input_pil = _generate_images(height=8, width=8, batch_size=1, input_type="pil")

        for output_type in ["np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_pil), output_type=output_type)
            for i, o in zip(input_pil, out):
                in_np = np.array(i)
                out_np = to_np(out) if output_type == "pil" else (to_np(out) * 255).round()
                self.assertTrue(np.allclose(in_np, out_np, atol=1e-6))
