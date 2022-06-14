import os
import unittest

import torch


def require_hf_token(test_case):
    """
    Decorator marking a test that requires huggingface hub token.
    """
    use_auth_token = os.environ.get("HF_AUTH_TOKEN", None)
    if use_auth_token is None:
        return unittest.skip("test requires hf token as `HF_AUTH_TOKEN` environment variable")(test_case)
    else:
        return test_case


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    return unittest.skipUnless(torch_device == "cuda", "test requires CUDA")(test_case)
