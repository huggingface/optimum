import random
import tempfile
import unittest

import requests as r
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import TOKEN, is_staging_test

from optimum.modeling_base import OptimizedModel


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


@is_staging_test
class TestOptimizedModel(unittest.TestCase):
    def test_load_model_from_hub(self):
        dummy_model = DummyModel.from_pretrained(TEST_HUB_PATH)
        self.assertTrue(dummy_model.config.remote)

    def test_push_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = DummyModel.from_pretrained(TEST_LOCAL_PATH)
            # create remote hash to check if file was updated.
            remote_hash = random.getrandbits(128)
            model.config.from_local = remote_hash

            model.save_pretrained(tmpdirname, push_to_hub=True, token=TOKEN)
            # folder contains all config files and pytorch_model.bin
            url = "https://huggingface.co/philschmid/unit_test_save_model/raw/main/config.json"
            response = r.get(url)
            self.assertEqual(remote_hash, response.json()["from_local"])
