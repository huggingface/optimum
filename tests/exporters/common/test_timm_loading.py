# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import importlib
import tempfile
from unittest import TestCase, mock

from optimum.exporters.tasks import TasksManager


def _load_timm_via_tasks_manager(model_name_or_path: str) -> mock.MagicMock:
    """Runs the timm loading branch of `TasksManager.get_model_from_task` with a fake timm module
    and returns the `create_model` mock, so the test can assert which source prefix was used."""
    fake_timm = mock.MagicMock()
    create_model = mock.MagicMock()
    fake_timm.create_model = create_model

    real_import_module = importlib.import_module

    def patched_import(name, *args, **kwargs):
        if name == "timm":
            return fake_timm
        return real_import_module(name, *args, **kwargs)

    with mock.patch("optimum.exporters.tasks.importlib.import_module", side_effect=patched_import):
        TasksManager.get_model_from_task(
            task="image-classification",
            model_name_or_path=model_name_or_path,
            framework="pt",
            library_name="timm",
        )
    return create_model


class TimmLocalDirLoadingTestCase(TestCase):
    def test_local_timm_path_uses_local_dir_prefix(self):
        # Regression test for #2423: a local directory must be loaded with the `local-dir:` prefix,
        # not `hf_hub:` (which timm would interpret as a Hub repo id and fail to download).
        with tempfile.TemporaryDirectory() as local_dir:
            create_model = _load_timm_via_tasks_manager(local_dir)
            create_model.assert_called_once_with(f"local-dir:{local_dir}", pretrained=True, exportable=True)

    def test_hub_timm_path_uses_hf_hub_prefix(self):
        # A non-local (Hub repo id) argument must keep using the `hf_hub:` prefix, as before.
        create_model = _load_timm_via_tasks_manager("timm/mobilenetv3_large_100.ra_in1k")
        create_model.assert_called_once_with(
            "hf_hub:timm/mobilenetv3_large_100.ra_in1k", pretrained=True, exportable=True
        )
