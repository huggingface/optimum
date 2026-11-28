# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from torch import nn
from torch.fx import symbolic_trace

from optimum.fx.optimization import FuseBatchNorm1dInLinear


def test_fuse_batchnorm_before_biasless_rectangular_linear():
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2, bias=False)).eval()
    with torch.no_grad():
        model[0].running_mean.copy_(torch.tensor([0.5, -1.0, 2.0]))
        model[0].running_var.copy_(torch.tensor([0.25, 4.0, 1.0]))
        model[0].weight.copy_(torch.tensor([1.5, 0.5, -2.0]))
        model[0].bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        model[1].weight.copy_(torch.tensor([[1.0, 2.0, -1.0], [-0.5, 1.0, 3.0]]))
        inputs = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
        expected = model(inputs)

        fused = FuseBatchNorm1dInLinear()(symbolic_trace(model))

        torch.testing.assert_close(fused(inputs), expected)
    assert not any(isinstance(module, nn.BatchNorm1d) for module in fused.modules())
    assert fused.get_submodule("1").bias.shape == (2,)
