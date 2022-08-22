#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gc
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

import onnx
from onnxruntime import InferenceSession
from optimum.onnxruntime import ORTConfig, ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
from optimum.onnxruntime.modeling_ort import ORTModelForSequenceClassification
from parameterized import parameterized


class ORTConfigTest(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            optimization_config = OptimizationConfig(optimization_level=2)
            ort_config = ORTConfig(opset=11, quantization=quantization_config, optimization=optimization_config)
            ort_config.save_pretrained(tmp_dir)
            loaded_ort_config = ORTConfig.from_pretrained(tmp_dir)
            self.assertEqual(ort_config.to_dict(), loaded_ort_config.to_dict())


class ORTOptimizerTest(unittest.TestCase):

    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = (
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-bert"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-distilbert"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-bart"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-gpt2"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-roberta"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-electra"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-xlm-roberta"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-big_bird"),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID)
    def test_compare_original_onnx_model_with_optimized_onnx_model(self, model_cls, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        optimization_config = OptimizationConfig(optimization_level=2, optimize_with_onnxruntime_only=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = model_cls.from_pretrained(model_name, from_transformers=True)
            model.save_pretrained(tmp_dir)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.fit(optimization_config=optimization_config, save_dir=tmp_dir)
            optimized_model = model_cls.from_pretrained(
                tmp_dir, file_name="model_optimized.onnx", from_transformers=False
            )
            tokens = tokenizer("This is a sample input", return_tensors="pt")
            model_outputs = model(**tokens)
            optimized_model_outputs = optimized_model(**tokens)

            # Compare tensors outputs
            self.assertTrue(torch.allclose(model_outputs.logits, optimized_model_outputs.logits, atol=1e-4))
            gc.collect()

    def test_optimization_details(self):
        model_name = "hf-internal-testing/tiny-random-distilbert"
        optimization_config = OptimizationConfig(optimization_level=0, optimize_with_onnxruntime_only=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
            model.save_pretrained(output_dir)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.fit(optimization_config=optimization_config, save_dir=output_dir)
            model_path = output_dir.joinpath("model.onnx")
            optimized_model_path = output_dir.joinpath("model_optimized.onnx")
            difference_nodes_number = optimizer.get_nodes_number_difference(model_path, optimized_model_path)
            fused_operator = optimizer.get_fused_operators(model_path)
            sorted_operators_difference = optimizer.get_operators_difference(model_path, optimized_model_path)
            self.assertEqual(difference_nodes_number, 0)
            self.assertEqual(len(fused_operator), 0)
            self.assertEqual(len(sorted_operators_difference), 0)
            gc.collect()

    def test_optimization_fp16(self):
        model_name = "hf-internal-testing/tiny-random-distilbert"
        optimization_config = OptimizationConfig(optimization_level=0, fp16=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
            model.save_pretrained(tmp_dir)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.fit(optimization_config=optimization_config, save_dir=tmp_dir)
            optimized_model = onnx.load(os.path.join(tmp_dir, "model_optimized.onnx"))
            for w in optimized_model.graph.initializer:
                self.assertNotEqual(w.data_type, onnx.onnx_pb.TensorProto.FLOAT)

            optimized_model = ORTModelForSequenceClassification.from_pretrained(
                tmp_dir, file_name="model_optimized.onnx", from_transformers=False
            )
            tokens = tokenizer("This is a sample input", return_tensors="pt")
            model_outputs = model(**tokens)
            optimized_model_outputs = optimized_model(**tokens)

            # Compare tensors outputs
            self.assertTrue(torch.allclose(model_outputs.logits, optimized_model_outputs.logits, atol=1e-4))
