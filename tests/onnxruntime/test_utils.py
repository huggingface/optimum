import unittest

import torch

from optimum.onnxruntime.utils import get_device_for_provider, get_provider_for_device


class TestProviderAndDeviceGetters(unittest.TestCase):
    def test_get_device_for_provider(self):
        self.assertEqual(get_device_for_provider("CPUExecutionProvider"), torch.device("cpu"))
        self.assertEqual(get_device_for_provider("CUDAExecutionProvider"), torch.device("cuda"))

    def test_get_provider_for_device(self):
        self.assertEqual(get_provider_for_device(torch.device("cpu")), "CPUExecutionProvider")
        self.assertEqual(get_provider_for_device(torch.device("cuda")), "CUDAExecutionProvider")
