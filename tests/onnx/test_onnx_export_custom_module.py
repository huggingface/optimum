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
from unittest import TestCase

import torch
from transformers.models.sew_d import modeling_sew_d


class StableDropoutTestCase(TestCase):
    """Tests export of StableDropout module."""

    def test_training(self):
        """Tests export of StableDropout in training mode."""

        devnull = open(os.devnull, "wb")
        # drop_prob must be > 0 for the test to be meaningful
        sd = modeling_sew_d.StableDropout(0.1)
        # Avoid warnings in training mode
        do_constant_folding = False
        # Dropout is a no-op in inference mode
        training = torch.onnx.TrainingMode.PRESERVE
        input = (torch.randn(2, 2),)

        # Expected to pass on torch >= 2.5
        torch.onnx.export(
            sd,
            input,
            devnull,
            opset_version=12,
            do_constant_folding=do_constant_folding,
            training=training,
        )

        devnull.close()

    def test_inference(self):
        """Tests export of StableDropout in inference mode."""

        devnull = open(os.devnull, "wb")
        # drop_prob must be > 0 for the test to be meaningful
        sd = modeling_sew_d.StableDropout(0.1)
        # Dropout is a no-op in inference mode
        training = torch.onnx.TrainingMode.EVAL
        input = (torch.randn(2, 2),)

        # Expected to pass on torch >= 2.5
        torch.onnx.export(
            sd,
            input,
            devnull,
            opset_version=12,
            do_constant_folding=True,
            training=training,
        )

        devnull.close()
