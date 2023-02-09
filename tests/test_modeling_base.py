import os
import random
import tempfile
import unittest

import requests as r
import torch
from transformers.configuration_utils import PretrainedConfig

from optimum.modeling_base import OptimizedModel
from optimum.utils.testing_utils import require_hf_token


TEST_HUB_PATH = "philschmid/unit_test_model"
TEST_LOCAL_PATH = "tests/assets/hub"


class DummyModel(OptimizedModel):
    def _save_pretrained(self, save_directory, **kwargs):
        return

    @classmethod
    def _from_pretrained(cls, **kwargs):
        config = PretrainedConfig.from_dict(kwargs["config"])
        model = cls(model=torch.nn.Module, config=config)
        return model

    def forward(self, *args, **kwargs):
        pass


class TestOptimizedModel(unittest.TestCase):
    def test_load_model_from_hub(self):
        # TODO: figure out how to create repos and push stuff to staging
        if os.getenv("HUGGINGFACE_CO_STAGING", False):
            self.skipTest("Skip test on staging")

        dummy_model = DummyModel.from_pretrained(TEST_HUB_PATH)
        self.assertTrue(dummy_model.config.remote)

    @require_hf_token
    def test_push_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = DummyModel.from_pretrained(TEST_LOCAL_PATH)
            # create remote hash to check if file was updated.
            remote_hash = random.getrandbits(128)
            model.config.from_local = remote_hash

            model.save_pretrained(
                tmpdirname,
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                push_to_hub=True,
                repository_id="unit_test_save_model",
            )
            # folder contains all config files and pytorch_model.bin
            url = "https://huggingface.co/philschmid/unit_test_save_model/raw/main/config.json"
            response = r.get(url)
            self.assertEqual(remote_hash, response.json()["from_local"])
