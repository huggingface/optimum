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
import random
import unittest
from typing import Dict

import numpy as np
import pytest
import torch
from diffusers import (
    OnnxStableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import floats_tensor, load_image
from parameterized import parameterized
from transformers.testing_utils import require_torch_gpu
from utils_onnxruntime_tests import MODEL_NAMES, SEED, ORTModelTestMixin

from optimum.onnxruntime import ORTStableDiffusionPipeline
from optimum.onnxruntime.modeling_diffusion import (
    ORTModelTextEncoder,
    ORTModelUnet,
    ORTModelVaeDecoder,
    ORTModelVaeEncoder,
    ORTStableDiffusionImg2ImgPipeline,
    ORTStableDiffusionInpaintPipeline,
    ORTStableDiffusionXLPipeline,
)
from optimum.utils import logging
from optimum.utils.testing_utils import grid_parameters, require_diffusers


logger = logging.get_logger()


def _generate_inputs():
    inputs = {
        "prompt": "sailing ship in storm by Leonardo da Vinci",
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "np",
    }
    return inputs


class ORTStableDiffusionPipelineBase(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionPipeline
    TASK = "stable-diffusion"

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn(
            f"does not appear to have a file named {self.ORTMODEL_CLASS.config_name}", str(context.exception)
        )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        num_images_per_prompt = 4
        batch_size = 6
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        self.assertEqual(pipeline.vae_scale_factor, 2)
        self.assertEqual(pipeline.vae_decoder.config["latent_channels"], 4)
        self.assertEqual(pipeline.unet.config["in_channels"], 4)
        inputs = self.generate_inputs()
        outputs = pipeline(**inputs).images
        self.assertEqual(outputs.shape, (1, 128, 128, 3))
        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
        self.assertEqual(outputs.shape, (num_images_per_prompt, 128, 128, 3))
        outputs = pipeline([inputs.pop("prompt")] * batch_size, **inputs).images
        self.assertEqual(outputs.shape, (batch_size, 128, 128, 3))

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider"]})
    )
    @require_torch_gpu
    @pytest.mark.gpu_test
    @require_diffusers
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        inputs = self.generate_inputs()
        outputs = pipeline(**inputs).images
        # Verify model devices
        self.assertEqual(pipeline.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (1, 128, 128, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_callback(self, model_arch: str):
        def callback_fn(step: int, timestep: int, latents: np.ndarray) -> None:
            callback_fn.has_been_called = True
            callback_fn.number_of_steps += 1

        pipe = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)
        callback_fn.has_been_called = False
        callback_fn.number_of_steps = 0
        inputs = self.generate_inputs(height=64, width=64)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        self.assertTrue(callback_fn.has_been_called)
        self.assertEqual(callback_fn.number_of_steps, inputs["num_inference_steps"])

    def generate_inputs(self, height=128, width=128):
        inputs = _generate_inputs()
        inputs["height"] = height
        inputs["width"] = width
        return inputs


class ORTStableDiffusionImg2ImgPipelineTest(ORTStableDiffusionPipelineBase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionImg2ImgPipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        inputs = self.generate_inputs()
        inputs["prompt"] = "A painting of a squirrel eating a burger"

        output = pipeline(**inputs, generator=np.random.RandomState(0)).images[0, -3:, -3:, -1]
        # https://github.com/huggingface/diffusers/blob/v0.17.1/tests/pipelines/stable_diffusion/test_onnx_stable_diffusion_img2img.py#L71
        expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.58760, 0.55368, 0.59643, 0.51529, 0.41217, 0.49087])
        self.assertTrue(np.allclose(output.flatten(), expected_slice, atol=1e-1))

        # Verify it can be loaded with ORT diffusers pipeline
        diffusers_pipeline = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusers_output = diffusers_pipeline(**inputs, generator=np.random.RandomState(0)).images[0, -3:, -3:, -1]
        self.assertTrue(np.allclose(output, diffusers_output, atol=1e-4))

    def generate_inputs(self, height=128, width=128):
        inputs = _generate_inputs()
        inputs["image"] = floats_tensor((1, 3, height, width), rng=random.Random(SEED))
        inputs["strength"] = 0.75
        return inputs


class ORTStableDiffusionPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionPipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers(self, model_arch: str):
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)
        self.assertIsInstance(ort_pipeline.text_encoder, ORTModelTextEncoder)
        self.assertIsInstance(ort_pipeline.vae_decoder, ORTModelVaeDecoder)
        self.assertIsInstance(ort_pipeline.vae_encoder, ORTModelVaeEncoder)
        self.assertIsInstance(ort_pipeline.unet, ORTModelUnet)
        self.assertIsInstance(ort_pipeline.config, Dict)

        pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAMES[model_arch])
        pipeline.safety_checker = None
        num_images_per_prompt, height, width = 1, 64, 64
        latents_shape = (
            num_images_per_prompt,
            ort_pipeline.unet.config["in_channels"],
            height // ort_pipeline.vae_scale_factor,
            width // ort_pipeline.vae_scale_factor,
        )
        latents = np.random.randn(*latents_shape).astype(np.float32)
        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_rescale": 0.1,
        }

        for output_type in ["latent", "np"]:
            ort_outputs = ort_pipeline(latents=latents, output_type=output_type, **kwargs).images
            self.assertIsInstance(ort_outputs, np.ndarray)
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images
            # Compare model outputs
            self.assertTrue(np.allclose(ort_outputs, outputs, atol=1e-4))
            # Compare model devices
            self.assertEqual(pipeline.device, ort_pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        pipeline = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)
        inputs = _generate_inputs()
        height = 64
        width = 64
        np.random.seed(0)
        ort_outputs_1 = pipeline(**inputs, height=height, width=width)
        np.random.seed(0)
        ort_outputs_2 = pipeline(**inputs, height=height, width=width)
        ort_outputs_3 = pipeline(**inputs, height=height, width=width)
        # Compare model outputs
        self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
        self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))


class ORTStableDiffusionXLPipelineTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion-xl",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionXLPipeline
    TASK = "stable-diffusion-xl"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers(self, model_arch: str):
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)
        self.assertIsInstance(ort_pipeline.text_encoder, ORTModelTextEncoder)
        self.assertIsInstance(ort_pipeline.text_encoder_2, ORTModelTextEncoder)
        self.assertIsInstance(ort_pipeline.vae_decoder, ORTModelVaeDecoder)
        self.assertIsInstance(ort_pipeline.vae_encoder, ORTModelVaeEncoder)
        self.assertIsInstance(ort_pipeline.unet, ORTModelUnet)
        self.assertIsInstance(ort_pipeline.config, Dict)

        pipeline = StableDiffusionXLPipeline.from_pretrained(MODEL_NAMES[model_arch])
        # image size must be higher or equal to 256 to encode with watermarker
        batch_size, num_images_per_prompt, height, width = 1, 2, 256, 256
        latents = ort_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            ort_pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_rescale": 0.1,
        }

        for output_type in ["latent", "np"]:
            ort_outputs = ort_pipeline(latents=latents, output_type=output_type, **kwargs).images
            self.assertIsInstance(ort_outputs, np.ndarray)
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images
            # Compare model outputs
            self.assertTrue(np.allclose(ort_outputs, outputs, atol=1e-4))
            # Compare model devices
            self.assertEqual(pipeline.device, ort_pipeline.device)


class ORTStableDiffusionInpaintPipelineTest(ORTStableDiffusionPipelineBase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionInpaintPipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        height = 64
        width = 64
        latents_shape = (
            1,
            ort_pipeline.vae_decoder.config["latent_channels"],
            height // ort_pipeline.vae_scale_factor,
            width // ort_pipeline.vae_scale_factor,
        )
        latents = np.random.randn(*latents_shape).astype(np.float32)
        inputs = self.generate_inputs(height=height, width=width)
        outputs = ort_pipeline(**inputs, latents=latents).images
        self.assertEqual(outputs.shape, (1, height, width, 3))
        expected_slice = np.array([0.5442, 0.3002, 0.5665, 0.6485, 0.4421, 0.6441, 0.5778, 0.5076, 0.5612])
        self.assertTrue(np.allclose(outputs[0, -3:, -3:, -1].flatten(), expected_slice, atol=1e-4))

    def generate_inputs(self, height=128, width=128):
        inputs = super(ORTStableDiffusionInpaintPipelineTest, self).generate_inputs(height, width)
        inputs["image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        ).resize((64, 64))

        inputs["mask_image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        ).resize((64, 64))

        return inputs
