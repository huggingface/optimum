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

import torch
from transformers import AutoTokenizer

import onnx
from optimum.onnxruntime import ORTConfig, ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime.modeling_seq2seq import ORTModelForSeq2SeqLM
from parameterized import parameterized


class ORTOptimizerTest(unittest.TestCase):

    # Contribution note: Please add test models in alphabetical order. Find test models here: https://huggingface.co/hf-internal-testing.
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = (
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-bart"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-bert"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-big_bird"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-distilbert"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-electra"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-gpt2"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-roberta"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-xlm-roberta"),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID)
    def test_compare_original_model_with_optimized_model(self, model_cls, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        optimization_config = OptimizationConfig(optimization_level=2, enable_transformers_specific_optimizations=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = model_cls.from_pretrained(model_name, from_transformers=True)
            model.save_pretrained(tmp_dir)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.optimize(optimization_config=optimization_config, save_dir=tmp_dir)
            optimized_model = model_cls.from_pretrained(
                tmp_dir, file_name="model_optimized.onnx", from_transformers=False
            )
            expected_ort_config = ORTConfig(optimization=optimization_config)
            ort_config = ORTConfig.from_pretrained(tmp_dir)

            # Verify the ORTConfig was correctly created and saved
            self.assertEqual(ort_config.to_dict(), expected_ort_config.to_dict())

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            model_outputs = model(**tokens)
            optimized_model_outputs = optimized_model(**tokens)

            # Compare tensors outputs
            self.assertTrue(torch.allclose(model_outputs.logits, optimized_model_outputs.logits, atol=1e-4))
            gc.collect()

    # Contribution note: Please add test models in alphabetical order. Find test models here: https://huggingface.co/hf-internal-testing.
    SUPPORTED_SEQ2SEQ_ARCHITECTURES_WITH_MODEL_ID = (
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-bart", False),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-bart", True),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-LongT5ForConditionalGeneration", False),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-LongT5ForConditionalGeneration", True),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-marian", False),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-marian", True),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-mbart", False),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-mbart", True),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-onnx-mt5", False),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-onnx-mt5", True),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-m2m_100", False),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-m2m_100", True),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-pegasus", False),
        (ORTModelForSeq2SeqLM, "hf-internal-testing/tiny-random-pegasus", True),
    )

    @parameterized.expand(SUPPORTED_SEQ2SEQ_ARCHITECTURES_WITH_MODEL_ID)
    def test_compare_original_seq2seq_model_with_optimized_model(self, model_cls, model_name, use_cache):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        optimization_config = OptimizationConfig(optimization_level=2, enable_transformers_specific_optimizations=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = model_cls.from_pretrained(model_name, from_transformers=True, use_cache=use_cache)
            model.save_pretrained(tmp_dir)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.optimize(optimization_config=optimization_config, save_dir=tmp_dir)
            optimized_model = model_cls.from_pretrained(
                tmp_dir,
                from_transformers=False,
                use_cache=use_cache,
            )

            expected_ort_config = ORTConfig(optimization=optimization_config)
            ort_config = ORTConfig.from_pretrained(tmp_dir)

            # Verify the ORTConfig was correctly created and saved
            self.assertEqual(ort_config.to_dict(), expected_ort_config.to_dict())

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            model_outputs = model.generate(**tokens)
            optimized_model_outputs = optimized_model.generate(**tokens)
            # Compare tensors outputs
            self.assertTrue(torch.equal(model_outputs, optimized_model_outputs))
            gc.collect()

    def test_optimization_details(self):
        model_name = "hf-internal-testing/tiny-random-distilbert"
        optimization_config = OptimizationConfig(
            optimization_level=0, enable_transformers_specific_optimizations=False
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
            model.save_pretrained(output_dir)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.optimize(optimization_config=optimization_config, save_dir=output_dir)
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
            optimizer.optimize(optimization_config=optimization_config, save_dir=tmp_dir)
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
