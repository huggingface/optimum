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
from utils_onnxruntime_tests import MODEL_NAMES, SEED

from optimum.exporters import TasksManager
from optimum.onnxruntime import (
    ONNX_WEIGHTS_NAME,
    ORTStableDiffusionPipeline,
)
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


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class ORTModelTestMixin(unittest.TestCase):
    ARCH_MODEL_MAP = {}

    TENSOR_ALIAS_TO_TYPE = {
        "pt": torch.Tensor,
        "np": np.ndarray,
    }

    @classmethod
    def setUpClass(cls):
        cls.onnx_model_dirs = {}

    def _setup(self, model_args: Dict):
        """
        Exports the PyTorch models to ONNX ahead of time to avoid multiple exports during the tests.
        We don't use unittest setUpClass, in order to still be able to run individual tests.
        """
        model_arch = model_args["model_arch"]
        model_arch_and_params = model_args["test_name"]

        # TODO: this should actually be checked in ORTModel!
        task = self.TASK
        if "use_cache" in model_args and model_args["use_cache"] is True:
            task = task + "-with-past"

        if "use_cache" in model_args and task not in TasksManager.get_supported_tasks_for_model_type(
            model_arch.replace("_", "-"), exporter="onnx"
        ):
            self.skipTest("Unsupported export case")

        if model_arch_and_params not in self.onnx_model_dirs:
            # model_args will contain kwargs to pass to ORTModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")

            model_id = (
                self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
            )
            set_seed(SEED)
            onnx_model = self.ORTMODEL_CLASS.from_pretrained(model_id, **model_args, export=True)

            model_dir = tempfile.mkdtemp(prefix=f"{model_arch_and_params}_{self.TASK}_")
            onnx_model.save_pretrained(model_dir)
            self.onnx_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.onnx_model_dirs.items():
            shutil.rmtree(dir_path)


class ORTModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID = "hf-internal-testing/tiny-random-OnnxStableDiffusionPipeline"

    def test_load_stable_diffusion_model_from_cache(self):
        _ = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)  # caching

        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, local_files_only=True
        )

        self.assertIsInstance(model.text_encoder, ORTModelTextEncoder)
        self.assertIsInstance(model.vae_decoder, ORTModelVaeDecoder)
        self.assertIsInstance(model.vae_encoder, ORTModelVaeEncoder)
        self.assertIsInstance(model.unet, ORTModelUnet)
        self.assertIsInstance(model.config, Dict)

    def test_load_stable_diffusion_model_from_empty_cache(self):
        dirpath = os.path.join(
            default_cache_path, "models--" + self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID.replace("/", "--")
        )

        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        with self.assertRaises(Exception):
            _ = ORTStableDiffusionPipeline.from_pretrained(
                self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, local_files_only=True
            )

    def test_load_stable_diffusion_model_from_hub(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        self.assertIsInstance(model.text_encoder, ORTModelTextEncoder)
        self.assertIsInstance(model.vae_decoder, ORTModelVaeDecoder)
        self.assertIsInstance(model.vae_encoder, ORTModelVaeEncoder)
        self.assertIsInstance(model.unet, ORTModelUnet)
        self.assertIsInstance(model.config, Dict)

    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_load_stable_diffusion_model_cuda_provider(self):
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="CUDAExecutionProvider"
        )
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.unet.session.get_providers(), model.providers)
        self.assertListEqual(model.text_encoder.session.get_providers(), model.providers)
        self.assertListEqual(model.vae_decoder.session.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    def test_load_stable_diffusion_model_cpu_provider(self):
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="CPUExecutionProvider"
        )
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])
        self.assertListEqual(model.unet.session.get_providers(), model.providers)
        self.assertListEqual(model.text_encoder.session.get_providers(), model.providers)
        self.assertListEqual(model.vae_decoder.session.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cpu"))

    def test_load_stable_diffusion_model_unknown_provider(self):
        with self.assertRaises(ValueError):
            ORTStableDiffusionPipeline.from_pretrained(
                self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="FooExecutionProvider"
            )

    def test_passing_session_options_stable_diffusion(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 3
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, session_options=options
        )
        self.assertEqual(model.unet.session.get_session_options().intra_op_num_threads, 3)
        self.assertEqual(model.text_encoder.session.get_session_options().intra_op_num_threads, 3)
        self.assertEqual(model.vae_decoder.session.get_session_options().intra_op_num_threads, 3)

    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_passing_provider_options_stable_diffusion(self):
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="CUDAExecutionProvider"
        )
        self.assertEqual(
            model.unet.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1"
        )
        self.assertEqual(
            model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"],
            "1",
        )
        self.assertEqual(
            model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1"
        )

        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID,
            provider="CUDAExecutionProvider",
            provider_options={"do_copy_in_default_stream": 0},
        )
        self.assertEqual(
            model.unet.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0"
        )
        self.assertEqual(
            model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"],
            "0",
        )
        self.assertEqual(
            model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0"
        )

    def test_stable_diffusion_model_on_cpu(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.unet.device, cpu)
        self.assertEqual(model.text_encoder.device, cpu)
        self.assertEqual(model.vae_decoder.device, cpu)
        self.assertEqual(model.unet.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    # test string device input for to()
    def test_stable_diffusion_model_on_cpu_str(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        cpu = torch.device("cpu")
        model.to("cpu")
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.unet.device, cpu)
        self.assertEqual(model.text_encoder.device, cpu)
        self.assertEqual(model.vae_decoder.device, cpu)
        self.assertEqual(model.unet.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_stable_diffusion_model_on_gpu(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.device, torch.device("cuda:0"))
        self.assertEqual(model.text_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @unittest.skipIf(get_gpu_count() <= 1, "this test requires multi-gpu")
    def test_stable_diffusion_model_on_gpu_id(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to(torch.device("cuda:1"))
        self.assertEqual(model.unet.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to(1)
        self.assertEqual(model.unet.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to("cuda:1")
        self.assertEqual(model.unet.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

    # test string device input for to()
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_stable_diffusion_model_on_gpu_str(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.device, torch.device("cuda:0"))
        self.assertEqual(model.text_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_save_stable_diffusion_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
            model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(model.config_name, folder_contents)
            for subfoler in {
                DIFFUSION_MODEL_UNET_SUBFOLDER,
                DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
            }:
                folder_contents = os.listdir(os.path.join(tmpdirname, subfoler))
                self.assertIn(ONNX_WEIGHTS_NAME, folder_contents)


class ORTStableDiffusionPipelineIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ORTMODEL_CLASS = ORTStableDiffusionPipeline
    TASK = "stable-diffusion"

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTStableDiffusionPipeline.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn(
            f"does not appear to have a file named {ORTStableDiffusionPipeline.config_name}", str(context.exception)
        )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        ort_pipeline = ORTStableDiffusionPipeline.from_pretrained(self.onnx_model_dirs[model_arch])

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
    def test_num_images_per_prompt(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        num_images_per_prompt = 4
        batch_size = 6

        pipeline = ORTStableDiffusionPipeline.from_pretrained(self.onnx_model_dirs[model_arch])
        prompt = "sailing ship in storm by Leonardo da Vinci"
        outputs = pipeline(prompt, num_inference_steps=2, output_type="np").images
        self.assertEqual(outputs.shape, (1, 128, 128, 3))
        outputs = pipeline(
            prompt, num_inference_steps=2, num_images_per_prompt=num_images_per_prompt, output_type="np"
        ).images
        self.assertEqual(outputs.shape, (num_images_per_prompt, 128, 128, 3))
        outputs = pipeline([prompt] * batch_size, num_inference_steps=2, output_type="np").images
        self.assertEqual(outputs.shape, (batch_size, 128, 128, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        ort_pipeline = ORTStableDiffusionPipeline.from_pretrained(self.onnx_model_dirs[model_arch])
        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "output_type": "np",
            "num_inference_steps": 2,
        }
        np.random.seed(0)
        ort_outputs_1 = ort_pipeline(**kwargs)
        np.random.seed(0)
        ort_outputs_2 = ort_pipeline(**kwargs)
        ort_outputs_3 = ort_pipeline(**kwargs)

        # Compare model outputs
        self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
        self.assertFalse(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider"]})
    )
    @require_torch_gpu
    @pytest.mark.gpu_test
    @require_diffusers
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": test_name, "model_arch": model_arch}
        self._setup(model_args)
        pipe = ORTStableDiffusionPipeline.from_pretrained(self.onnx_model_dirs[test_name], provider=provider)
        outputs = pipe("sailing ship in storm by Leonardo da Vinci", output_type="np").images
        # Verify model devices
        self.assertEqual(pipe.device.type.lower(), "cuda")
        # Verify model outptus
        self.assertIsInstance(outputs, np.ndarray)
        self.assertEqual(outputs.shape, (1, 128, 128, 3))

    def _generate_random_data(self, input_image=False):
        from diffusers.utils import floats_tensor

        generator = np.random.RandomState(SEED)
        inputs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }

        if input_image:
            inputs["image"] = floats_tensor((1, 3, 128, 128), rng=random.Random(SEED))
            inputs["strength"] = 0.75

        return inputs
