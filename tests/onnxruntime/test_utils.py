import tempfile
import unittest

import onnxruntime as ort
import torch

from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig, ORTConfig
from optimum.onnxruntime.utils import get_device_for_provider, get_provider_for_device


class ProviderAndDeviceGettersTest(unittest.TestCase):
    def test_get_device_for_provider(self):
        self.assertEqual(get_device_for_provider("CPUExecutionProvider", provider_options={}), torch.device("cpu"))
        self.assertEqual(
            get_device_for_provider("CUDAExecutionProvider", provider_options={"device_id": 1}), torch.device("cuda:1")
        )

    def test_get_provider_for_device(self):
        self.assertEqual(get_provider_for_device(torch.device("cpu")), "CPUExecutionProvider")

        if "ROCMExecutionProvider" in ort.get_available_providers():
            self.assertEqual(get_provider_for_device(torch.device("cuda")), "ROCMExecutionProvider")
        else:
            self.assertEqual(get_provider_for_device(torch.device("cuda")), "CUDAExecutionProvider")


class ORTConfigTest(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            optimization_config = OptimizationConfig(optimization_level=2)
            ort_config = ORTConfig(opset=11, quantization=quantization_config, optimization=optimization_config)
            ort_config.save_pretrained(tmp_dir)
            loaded_ort_config = ORTConfig.from_pretrained(tmp_dir)
            self.assertEqual(ort_config.to_dict(), loaded_ort_config.to_dict())
