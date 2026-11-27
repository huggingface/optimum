# coding=utf-8
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
import pytest
import torch

from optimum.fx.optimization import MergeLinears


class Projections(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 3)
        self.k = torch.nn.Linear(4, 2)
        self.v = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)


@pytest.mark.parametrize("layout", ["root", "nested", "different_parents"])
def test_merge_linears_and_reverse(layout):
    torch.manual_seed(0)
    model = Projections().eval()
    if layout == "nested":
        model = torch.nn.Sequential(model)
    elif layout == "different_parents":
        for name in ("q", "k", "v"):
            setattr(model, name, torch.nn.Sequential(getattr(model, name)))
    inputs = torch.randn(2, 3, 4)
    expected = model(inputs)
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    traced = torch.fx.symbolic_trace(model)
    transformation = MergeLinears()

    transformed = transformation(traced)

    assert sum(isinstance(module, torch.nn.Linear) for module in transformed.modules()) == 1
    torch.testing.assert_close(transformed(inputs), expected)

    restored = transformation(transformed, reverse=True)

    assert sum(isinstance(module, torch.nn.Linear) for module in restored.modules()) == 3
    torch.testing.assert_close(restored(inputs), expected)
    torch.testing.assert_close(restored.state_dict(), original_state, rtol=0, atol=0)
