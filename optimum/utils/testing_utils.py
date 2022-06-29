import importlib.util
import os
import subprocess
import sys
import unittest


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
    if use_auth_token or has_sigopt_project is None:
        return unittest.skip("test requires an environment variable `SIGOPT_API_TOKEN` and `SIGOPT_PROJECT`")(
            test_case
        )
    else:
        return test_case


def require_ort_training(test_case):
    """
    Decorator marking a test that requires onnxruntime-training and torch_ort correctly installed and configured.
    These tests are skipped otherwise.
    """
    is_ort_train_available = importlib.util.find_spec("onnxruntime.training") is not None

    if importlib.util.find_spec("torch_ort") is not None:
        try:
            is_torch_ort_configured = True
            subprocess.run([sys.executable, "-m", "torch_ort.configure"], shell=False, check=True)
        except subprocess.CalledProcessError:
            is_torch_ort_configured = False

    return unittest.skipUnless(
        is_ort_train_available and is_torch_ort_configured,
        "test requires torch_ort correctly installed and configured",
    )(test_case)
