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
import PIL
import pytest
import torch
from diffusers import (
    OnnxStableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import floats_tensor
from packaging.version import Version, parse
from parameterized import parameterized
from transformers.testing_utils import require_torch_gpu
from utils_onnxruntime_tests import MODEL_NAMES, SEED, ORTModelTestMixin

from optimum.onnxruntime import (
    ORTLatentConsistencyModelPipeline,
    ORTStableDiffusionImg2ImgPipeline,
    ORTStableDiffusionInpaintPipeline,
    ORTStableDiffusionPipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
    ORTStableDiffusionXLPipeline,
)
from optimum.onnxruntime.modeling_diffusion import (
    ORTModelTextEncoder,
    ORTModelUnet,
    ORTModelVaeDecoder,
    ORTModelVaeEncoder,
)
from optimum.pipelines.diffusers.pipeline_utils import VaeImageProcessor
from optimum.utils.import_utils import _diffusers_version
from optimum.utils.testing_utils import grid_parameters, require_diffusers, require_ort_rocm


if parse(_diffusers_version) > Version("0.21.4"):
    from diffusers import LatentConsistencyModelPipeline


def _generate_inputs(batch_size=1):
    inputs = {
        "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "np",
    }
    return inputs


def _create_image(height=128, width=128, batch_size=1, channel=3, input_type="pil"):
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


class ORTStableDiffusionPipelineBase(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionPipeline
    TASK = "text-to-image"

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
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider"]})
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
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    @require_diffusers
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        height, width, batch_size = 128, 64, 1
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        if self.TASK == "image-to-image":
            input_types = ["np", "pil", "pt"]
        elif self.TASK == "text-to-image":
            input_types = ["np"]
        else:
            input_types = ["pil"]

        for input_type in input_types:
            if self.TASK == "image-to-image":
                inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, input_type=input_type)
            else:
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

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = _generate_inputs(batch_size=batch_size)
        inputs["height"] = height
        inputs["width"] = width
        return inputs


class ORTStableDiffusionImg2ImgPipelineTest(ORTStableDiffusionPipelineBase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionImg2ImgPipeline
    TASK = "image-to-image"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        height, width = 128, 128
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        inputs = self.generate_inputs(height=height, width=width)
        inputs["prompt"] = "A painting of a squirrel eating a burger"
        inputs["image"] = floats_tensor((1, 3, height, width), rng=random.Random(SEED))

        output = pipeline(**inputs, generator=np.random.RandomState(0)).images[0, -3:, -3:, -1]
        # https://github.com/huggingface/diffusers/blob/v0.17.1/tests/pipelines/stable_diffusion/test_onnx_stable_diffusion_img2img.py#L71
        expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.58760, 0.55368, 0.59643, 0.51529, 0.41217, 0.49087])
        self.assertTrue(np.allclose(output.flatten(), expected_slice, atol=1e-1))

        # Verify it can be loaded with ORT diffusers pipeline
        diffusers_pipeline = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.onnx_model_dirs[model_arch])
        diffusers_output = diffusers_pipeline(**inputs, generator=np.random.RandomState(0)).images[0, -3:, -3:, -1]
        self.assertTrue(np.allclose(output, diffusers_output, atol=1e-2))

    def generate_inputs(self, height=128, width=128, batch_size=1, input_type="np"):
        inputs = _generate_inputs(batch_size=batch_size)
        inputs["image"] = _create_image(height=height, width=width, batch_size=batch_size, input_type=input_type)
        inputs["strength"] = 0.75
        return inputs


class ORTStableDiffusionPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionPipeline
    TASK = "text-to-image"

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
        batch_size, num_images_per_prompt, height, width = 1, 2, 64, 32

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
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images

            self.assertIsInstance(ort_outputs, np.ndarray)
            # Compare model outputs
            self.assertTrue(np.allclose(ort_outputs, outputs, atol=1e-4))
            # Compare model devices
            self.assertEqual(pipeline.device, ort_pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        pipeline = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)
        inputs = _generate_inputs()
        height, width = 64, 32
        np.random.seed(0)
        ort_outputs_1 = pipeline(**inputs, height=height, width=width)
        np.random.seed(0)
        ort_outputs_2 = pipeline(**inputs, height=height, width=width)
        ort_outputs_3 = pipeline(**inputs, height=height, width=width)
        # Compare model outputs
        self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
        self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_negative_prompt(self, model_arch: str):
        pipeline = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)
        inputs = _generate_inputs()
        inputs["height"], inputs["width"] = 64, 32
        negative_prompt = ["This is a negative prompt"]
        np.random.seed(0)
        image_slice_1 = pipeline(**inputs, negative_prompt=negative_prompt).images[0, -3:, -3:, -1]
        prompt = inputs.pop("prompt")
        embeds = []
        for p in [prompt, negative_prompt]:
            text_inputs = pipeline.tokenizer(
                p,
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_inputs = text_inputs["input_ids"].astype(pipeline.text_encoder.input_dtype.get("input_ids", np.int32))
            embeds.append(pipeline.text_encoder(text_inputs)[0])

        inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = embeds
        np.random.seed(0)
        image_slice_2 = pipeline(**inputs).images[0, -3:, -3:, -1]
        self.assertTrue(np.allclose(image_slice_1, image_slice_2, atol=1e-4))


class ORTStableDiffusionXLPipelineTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion-xl",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionXLPipeline
    TASK = "text-to-image"

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
        batch_size, num_images_per_prompt, height, width = 2, 2, 64, 32
        latents = ort_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            ort_pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        kwargs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
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
        height, width = 64, 32
        np.random.seed(0)
        ort_outputs_1 = pipeline(**inputs, height=height, width=width)
        np.random.seed(0)
        ort_outputs_2 = pipeline(**inputs, height=height, width=width)
        ort_outputs_3 = pipeline(**inputs, height=height, width=width)
        self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
        self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))


class ORTStableDiffusionInpaintPipelineTest(ORTStableDiffusionPipelineBase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionInpaintPipeline
    TASK = "inpainting"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_diffusers_pipeline(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        height, width = 64, 64
        latents_shape = (
            1,
            ort_pipeline.vae_decoder.config["latent_channels"],
            height // ort_pipeline.vae_scale_factor,
            width // ort_pipeline.vae_scale_factor,
        )
        latents = np.random.randn(*latents_shape).astype(np.float32)
        inputs = self.generate_inputs(height=height, width=width)
        inputs["image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        ).resize((width, height))

        inputs["mask_image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        ).resize((width, height))

        outputs = ort_pipeline(**inputs, latents=latents).images
        self.assertEqual(outputs.shape, (1, height, width, 3))
        expected_slice = np.array([0.5442, 0.3002, 0.5665, 0.6485, 0.4421, 0.6441, 0.5778, 0.5076, 0.5612])
        self.assertTrue(np.allclose(outputs[0, -3:, -3:, -1].flatten(), expected_slice, atol=1e-4))

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = super(ORTStableDiffusionInpaintPipelineTest, self).generate_inputs(height, width)
        inputs["image"] = _create_image(height=height, width=width, batch_size=1, input_type="pil")[0]
        inputs["mask_image"] = _create_image(height=height, width=width, batch_size=1, input_type="pil")[0]
        return inputs


class ORTStableDiffusionXLImg2ImgPipelineTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion-xl",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionXLImg2ImgPipeline
    TASK = "image-to-image"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_inference(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        pipeline = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])

        height, width = 128, 128
        inputs = self.generate_inputs(height=height, width=width)
        inputs["image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        ).resize((width, height))
        output = pipeline(**inputs, generator=np.random.RandomState(0)).images[0, -3:, -3:, -1]
        expected_slice = np.array([0.6515, 0.5405, 0.4858, 0.5632, 0.5174, 0.5681, 0.4948, 0.4253, 0.5080])

        self.assertTrue(np.allclose(output.flatten(), expected_slice, atol=1e-1))

    def generate_inputs(self, height=128, width=128, batch_size=1, input_type="np"):
        inputs = _generate_inputs(batch_size=batch_size)
        inputs["image"] = _create_image(height=height, width=width, batch_size=batch_size, input_type=input_type)
        inputs["strength"] = 0.75
        return inputs


class ImageProcessorTest(unittest.TestCase):
    def test_vae_image_processor_pt(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
        input_pt = torch.stack(_create_image(height=8, width=8, batch_size=1, input_type="pt"))
        input_np = to_np(input_pt)

        for output_type in ["np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_pt), output_type=output_type)
            out_np = to_np(out)
            in_np = (input_np * 255).round() if output_type == "pil" else input_np
            self.assertTrue(np.allclose(in_np, out_np, atol=1e-6))

    def test_vae_image_processor_np(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
        input_np = np.stack(_create_image(height=8, width=8, input_type="np"))
        for output_type in ["np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_np), output_type=output_type)
            out_np = to_np(out)
            in_np = (input_np * 255).round() if output_type == "pil" else input_np
            self.assertTrue(np.allclose(in_np, out_np, atol=1e-6))

    def test_vae_image_processor_pil(self):
        image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
        input_pil = _create_image(height=8, width=8, batch_size=1, input_type="pil")

        for output_type in ["np", "pil"]:
            out = image_processor.postprocess(image_processor.preprocess(input_pil), output_type=output_type)
            for i, o in zip(input_pil, out):
                in_np = np.array(i)
                out_np = to_np(out) if output_type == "pil" else (to_np(out) * 255).round()
                self.assertTrue(np.allclose(in_np, out_np, atol=1e-6))


class ORTLatentConsistencyModelPipelineTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "latent-consistency",
    ]
    ORTMODEL_CLASS = ORTLatentConsistencyModelPipeline
    TASK = "text-to-image"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    @unittest.skipIf(
        parse(_diffusers_version) <= Version("0.21.4"),
        "not supported with this diffusers version, needs diffusers>=v0.22.0",
    )
    def test_compare_to_diffusers(self, model_arch: str):
        ort_pipeline = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)
        self.assertIsInstance(ort_pipeline.text_encoder, ORTModelTextEncoder)
        self.assertIsInstance(ort_pipeline.vae_decoder, ORTModelVaeDecoder)
        self.assertIsInstance(ort_pipeline.vae_encoder, ORTModelVaeEncoder)
        self.assertIsInstance(ort_pipeline.unet, ORTModelUnet)
        self.assertIsInstance(ort_pipeline.config, Dict)

        pipeline = LatentConsistencyModelPipeline.from_pretrained(MODEL_NAMES[model_arch])
        batch_size, num_images_per_prompt, height, width = 2, 2, 64, 32
        latents = ort_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            ort_pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        kwargs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_scale": 8.5,
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
