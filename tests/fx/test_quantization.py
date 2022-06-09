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
import torch
from torch.ao.quantization.quantize_fx import fuse_fx as orig_fuse_fx
from torch.ao.quantization.quantize_fx import prepare_fx as orig_prepare_fx
from torch.ao.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx
from transformers import PretrainedConfig

from optimum.fx.quantization import fuse_fx, prepare_fx, prepare_qat_fx


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(100, 10)
        self.linear1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 20)
        self.softmax = torch.nn.Softmax(dim=-1)

        # transformers.utils.fx.HFTracer needs those.
        self.device = torch.device("cpu")
        self.config = PretrainedConfig(hidden_size=10)

    def forward(self, input_ids: torch.Tensor):
        x = self.emb(input_ids)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = torch.matmul(x, x.transpose(0, 1))
        x = self.softmax(x)
        return x


def test_fuse_fx():
    model = DummyModel()
    model.eval()
    torch_fx_fused_model = orig_fuse_fx(model)
    optimum_fused_model = fuse_fx(model, input_names=["input_ids"], check=False)
    assert torch_fx_fused_model.code == optimum_fused_model.code


def test_prepare_fx():
    model = DummyModel()
    model.eval()
    qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
    torch_fx_prepared_model = orig_prepare_fx(model, qconfig_dict)
    optimum_prepared_model = prepare_fx(model, qconfig_dict, input_names=["input_ids"], check=False)
    assert torch_fx_prepared_model.code == optimum_prepared_model.code


def test_prepare_qat_fx():
    model = DummyModel()
    model.train()
    qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
    torch_fx_prepared_model = orig_prepare_qat_fx(model, qconfig_dict)
    optimum_prepared_model = prepare_qat_fx(model, qconfig_dict, input_names=["input_ids"], check=False)
    assert torch_fx_prepared_model.code == optimum_prepared_model.code
