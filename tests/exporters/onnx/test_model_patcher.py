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
from transformers import AutoModelForSequenceClassification

from optimum.exporters.onnx.model_configs import BertOnnxConfig
from optimum.exporters.onnx.model_patcher import ModelPatcher


def test_reset_unmodified_forward():
    model_checkpoint = "hf-internal-testing/tiny-random-bert"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    export_config = BertOnnxConfig(model.config)

    assert "forward" not in model.__dict__
    patcher = ModelPatcher(export_config, model)
    with patcher:
        assert "forward" in model.__dict__
        assert model.forward == patcher.patched_forward
    # Expected `forward` to be removed after patching context exits
    assert "forward" not in model.__dict__


def test_reset_overwritten_forward():
    model_checkpoint = "hf-internal-testing/tiny-random-bert"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    export_config = BertOnnxConfig(model.config)

    def foo(x):
        return x

    model.forward = foo

    assert model.__dict__["forward"] == foo
    patcher = ModelPatcher(export_config, model)
    with patcher:
        assert model.__dict__["forward"] != foo
        assert model.forward == patcher.patched_forward
    # Expected `forward` to be restored to the original overridden version after patching.
    assert model.__dict__["forward"] == foo
