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
import os
import pytest
from unittest import TestCase

from transformers import is_torch_available
from transformers.testing_utils import require_torch

if is_torch_available():
    import torch

    from transformers.models.deberta import modeling_deberta


class StableDropoutTestCase(TestCase):
    """Tests export of StableDropout module."""

    @require_torch
    @pytest.mark.filterwarnings("ignore:.*Dropout.*:UserWarning:torch.onnx.*")  # torch.onnx is spammy.
    def test_training(self):
        """Tests export of StableDropout in training mode."""
        devnull = open(os.devnull, "wb")
        # drop_prob must be > 0 for the test to be meaningful
        sd = modeling_deberta.StableDropout(0.1)
        # Avoid warnings in training mode
        do_constant_folding = False
        # Dropout is a no-op in inference mode
        training = torch.onnx.TrainingMode.PRESERVE
        input = (torch.randn(2, 2),)

        torch.onnx.export(
            sd,
            input,
            devnull,
            opset_version=12,  # Minimum supported
            do_constant_folding=do_constant_folding,
            training=training,
        )

        # Expected to fail with opset_version < 12
        with self.assertRaises(Exception):
            torch.onnx.export(
                sd,
                input,
                devnull,
                opset_version=11,
                do_constant_folding=do_constant_folding,
                training=training,
            )
