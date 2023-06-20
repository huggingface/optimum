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
import random
import shutil
import tempfile
import time
import unittest
from typing import Dict

import numpy as np
import onnxruntime
import pytest
import torch
from huggingface_hub.constants import default_cache_path
from parameterized import parameterized
from transformers import set_seed
from transformers.testing_utils import get_gpu_count, require_torch_gpu
from utils_onnxruntime_tests import MODEL_NAMES, SEED, ORTModelTestMixin

from optimum.exporters import TasksManager
from optimum.onnxruntime import (
    ONNX_WEIGHTS_NAME,
    ORTStableDiffusionPipeline,
)
from optimum.onnxruntime.modeling_diffusion import ORTStableDiffusionImg2ImgPipeline

from optimum.onnxruntime.modeling_diffusion import (
    ORTModelTextEncoder,
    ORTModelUnet,
    ORTModelVaeDecoder,
    ORTModelVaeEncoder,
)
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    logging,
)
from optimum.utils.testing_utils import grid_parameters, require_diffusers


logger = logging.get_logger()


def _generate_random_inputs(generate_image=False):
    from diffusers.utils import floats_tensor

    generator = np.random.RandomState(SEED)
    inputs = {
        "prompt": "sailing ship in storm by Leonardo da Vinci",
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "numpy",
    }

    if generate_image:
        inputs["image"] = floats_tensor((1, 3, 128, 128), rng=random.Random(SEED))
        inputs["strength"] = 0.75

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
        inputs = _generate_random_inputs(generate_image=isinstance(pipeline, ORTStableDiffusionImg2ImgPipeline))
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
        inputs = _generate_random_inputs(generate_image=isinstance(pipeline, ORTStableDiffusionImg2ImgPipeline))
        outputs = pipeline(**inputs).images
        # Verify model devices
        self.assertEqual(pipeline.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (1, 128, 128, 3))


class ORTStableDiffusionImg2ImgPipelineTest(ORTStableDiffusionPipelineBase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionImg2ImgPipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_stable_diffusion_pndm(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        inputs = _generate_random_inputs(generate_image=isinstance(pipeline, ORTStableDiffusionImg2ImgPipeline))
        inputs["prompt"] = "A painting of a squirrel eating a burger"

        output = pipeline(**inputs, generator=np.random.RandomState(0)).images[0, -3:, -3:, -1]
        # https://github.com/huggingface/diffusers/blob/v0.17.1/tests/pipelines/stable_diffusion/test_onnx_stable_diffusion_img2img.py#L71
        expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.58760, 0.55368, 0.59643, 0.51529, 0.41217, 0.49087])
        self.assertTrue(np.abs(expected_slice - output.flatten()).max() < 1e-1)


class ORTStableDiffusionPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        ort_pipeline = ORTStableDiffusionPipeline.from_pretrained(MODEL_NAMES[model_arch], export=True)
        self.assertIsInstance(ort_pipeline.text_encoder, ORTModelTextEncoder)
        self.assertIsInstance(ort_pipeline.vae_decoder, ORTModelVaeDecoder)
        self.assertIsInstance(ort_pipeline.vae_encoder, ORTModelVaeEncoder)
        self.assertIsInstance(ort_pipeline.unet, ORTModelUnet)
        self.assertIsInstance(ort_pipeline.config, Dict)

        from diffusers import StableDiffusionPipeline

        diffusers_pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        diffusers_pipeline.safety_checker = None
        num_images_per_prompt, height, width, scale_factor = 1, 512, 512, 8
        latents_shape = (
            num_images_per_prompt,
            diffusers_pipeline.unet.in_channels,
            height // scale_factor,
            width // scale_factor,
        )
        latents = np.random.randn(*latents_shape).astype(np.float32)
        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "num_inference_steps": 1,
            "output_type": "np",
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
        }
        ort_outputs = ort_pipeline(latents=latents, **kwargs).images
        self.assertIsInstance(ort_outputs, np.ndarray)

        with torch.no_grad():
            diffusers_outputs = diffusers_pipeline(latents=torch.from_numpy(latents), **kwargs).images
        # Compare model outputs
        self.assertTrue(np.allclose(ort_outputs, diffusers_outputs, atol=1e-4))
        # Compare model devices
        self.assertEqual(diffusers_pipeline.device, ort_pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        pipeline = ORTStableDiffusionPipeline.from_pretrained(MODEL_NAMES[model_arch])
        inputs = _generate_random_inputs()
        np.random.seed(0)
        ort_outputs_1 = pipeline(**inputs)
        np.random.seed(0)
        ort_outputs_2 = pipeline(**inputs)
        ort_outputs_3 = pipeline(**inputs)
        # Compare model outputs
        self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
        self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))
