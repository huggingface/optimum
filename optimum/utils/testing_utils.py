import importlib.util
import os
import subprocess
import sys
import unittest

from packaging import version

from optimum.utils import is_accelerate_available


def require_accelerate(test_case):
    """
    Decorator marking a test that requires accelerate. These tests are skipped when accelerate isn't installed.
    """
    return unittest.skipUnless(is_accelerate_available(), "test requires accelerate")(test_case)


def is_torch_greater_than_113():
    import torch

    return version.parse(torch.__version__) >= version.parse("1.13.0")


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    import torch

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    return unittest.skipUnless(torch_device == "cuda", "test requires CUDA")(test_case)


def require_hf_token(test_case):
    """
    Decorator marking a test that requires huggingface hub token.
    """
    use_auth_token = os.environ.get("HF_AUTH_TOKEN", None)
    if use_auth_token is None:
        return unittest.skip("test requires hf token as `HF_AUTH_TOKEN` environment variable")(test_case)
    else:
        return test_case


def require_sigopt_token_and_project(test_case):
    """
    Decorator marking a test that requires sigopt API token.
    """
    use_auth_token = os.environ.get("SIGOPT_API_TOKEN", None)
    has_sigopt_project = os.environ.get("SIGOPT_PROJECT", None)
    if use_auth_token is None or has_sigopt_project is None:
        return unittest.skip("test requires an environment variable `SIGOPT_API_TOKEN` and `SIGOPT_PROJECT`")(
            test_case
        )
    else:
        return test_case


def is_ort_training_available():
    is_ort_train_available = importlib.util.find_spec("onnxruntime.training") is not None

    if importlib.util.find_spec("torch_ort") is not None:
        try:
            is_torch_ort_configured = True
            subprocess.run([sys.executable, "-m", "torch_ort.configure"], shell=False, check=True)
        except subprocess.CalledProcessError:
            is_torch_ort_configured = False

    return is_ort_train_available and is_torch_ort_configured


def require_ort_training(test_case):
    """
    Decorator marking a test that requires onnxruntime-training and torch_ort correctly installed and configured.
    These tests are skipped otherwise.
    """
    return unittest.skipUnless(
        is_ort_training_available(),
        "test requires torch_ort correctly installed and configured",
    )(test_case)


def convert_to_hf_classes(mapping_dict):
    r"""
    Utility function useful in the context of testing `BetterTransformers` integration.
    """
    import transformers

    hf_names_dict = {}
    for fast_layer_key in mapping_dict.keys():
        if fast_layer_key == "TransformerBlock":
            # Hardcode it for distilbert - see https://github.com/huggingface/transformers/pull/19966
            prefix = "DistilBert"
        # For enc-decoder models the prefix is different
        elif "EncoderLayer" in fast_layer_key:
            prefix = fast_layer_key[:-12]
        else:
            prefix = fast_layer_key[:-5]

        # some `PreTrainedModel` models are not registerd in the auto mapping
        if hasattr(transformers, prefix + "PreTrainedModel"):
            hf_class = getattr(transformers, prefix + "PreTrainedModel")
        else:
            hf_class = getattr(transformers, prefix + "Model")

        hf_names_dict[fast_layer_key] = hf_class
    return hf_names_dict
