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
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional

import onnx
import pytest
import torch
from parameterized import parameterized
from transformers import AutoTokenizer
from transformers.testing_utils import require_torch_gpu
from utils_onnxruntime_tests import MODEL_NAMES

from optimum.exporters import TasksManager
from optimum.onnxruntime import AutoOptimizationConfig, ORTConfig, ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime.modeling_decoder import ORTModelForCausalLM
from optimum.onnxruntime.modeling_seq2seq import ORTModelForSeq2SeqLM
from optimum.utils.testing_utils import grid_parameters


class ORTOptimizerTestMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.onnx_model_dirs = {}

    def _setup(self, model_args: Dict):
        """
        Exports the PyTorch models to ONNX ahead of time to avoid multiple exports during the tests.
        We don't use unittest setUpClass, in order to still be able to run individual tests.
        """
        model_arch = model_args["model_arch"]
        model_arch_and_params = model_args["test_name"]

        # TODO: this should actually be checked in ORTModel!
        task = self.TASK
        if "use_cache" in model_args and model_args["use_cache"] is True:
            task = task + "-with-past"

        if "use_cache" in model_args and task not in TasksManager.get_supported_tasks_for_model_type(
            model_arch.replace("_", "-"), exporter="onnx"
        ):
            self.skipTest("Unsupported export case")

        if model_arch_and_params not in self.onnx_model_dirs:
            # model_args will contain kwargs to pass to ORTModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")

            model_id = MODEL_NAMES[model_arch]
            onnx_model = self.ORTMODEL_CLASS.from_pretrained(model_id, **model_args, from_transformers=True)

            model_dir = tempfile.mkdtemp(prefix=f"{model_arch_and_params}_{self.TASK}_")
            onnx_model.save_pretrained(model_dir)
            self.onnx_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.onnx_model_dirs.items():
            shutil.rmtree(dir_path)


class ORTOptimizerTest(unittest.TestCase):
    # Contribution note: Please add test models in alphabetical order. Find test models here: https://huggingface.co/hf-internal-testing.
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = (
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-bart"),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-bert"),
        # (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-big_bird"),
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


class ORTOptimizerForSeq2SeqLMIntegrationTest(ORTOptimizerTestMixin):
    TASK = "seq2seq-lm"
    ORTMODEL_CLASS = ORTModelForSeq2SeqLM

    SUPPORTED_ARCHITECTURES = [
        "bart",
        "blenderbot",
        "blenderbot_small",
        # "longt5",
        "m2m_100",
        "marian",
        "mbart",
        "mt5",
        "pegasus",
        "t5",
    ]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
    }

    def _test_optimization_levels(
        self,
        test_name: str,
        model_arch: str,
        use_cache: bool,
        optimization_level: str,
        provider: str,
        use_io_binding: Optional[bool] = None,
    ):
        export_name = test_name[:-3]  # remove `_OX` that is irrelevant as the export
        model_args = {"test_name": export_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        if provider == "CUDAExecutionProvider":
            for_gpu = True
            device = "cuda"
        else:
            for_gpu = False
            device = "cpu"

        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            self.onnx_model_dirs[export_name], use_cache=use_cache, provider=provider, use_io_binding=use_io_binding
        )

        optimizer = ORTOptimizer.from_pretrained(ort_model)

        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level, for_gpu=for_gpu)
        optimization_config.disable_shape_inference = True
        model_id = MODEL_NAMES[model_arch]

        with tempfile.TemporaryDirectory(suffix="_optimized") as tmp_dir:
            optimizer.optimize(save_dir=tmp_dir, optimization_config=optimization_config)

            optimized_model = ORTModelForSeq2SeqLM.from_pretrained(
                tmp_dir, use_cache=use_cache, provider=provider, use_io_binding=use_io_binding
            )

            expected_ort_config = ORTConfig(optimization=optimization_config)
            ort_config = ORTConfig.from_pretrained(tmp_dir)

            # Verify the ORTConfig was correctly created and saved
            self.assertEqual(ort_config.to_dict(), expected_ort_config.to_dict())

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokens = tokenizer("This is a sample input", return_tensors="pt").to(device)
            model_outputs = ort_model.generate(**tokens)
            optimized_model_outputs = optimized_model.generate(**tokens)

            self.assertTrue(torch.equal(model_outputs, optimized_model_outputs))
            gc.collect()

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [False, True],
                "optimization_level": ["O1", "O2", "O3"],
            }
        )
    )
    def test_optimization_levels_cpu(self, test_name: str, model_arch: str, use_cache: bool, optimization_level: str):
        self._test_optimization_levels(
            test_name=test_name,
            model_arch=model_arch,
            use_cache=use_cache,
            optimization_level=optimization_level,
            provider="CPUExecutionProvider",
        )

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "optimization_level": ["O1", "O2", "O3", "O4"],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_optimization_levels_gpu(self, test_name: str, model_arch: str, use_cache: bool, optimization_level: str):
        for use_io_binding in [False, True]:
            # TODO: investigate why marian with IO Binding fails
            if model_arch == "marian" and use_io_binding is True:
                continue

            self._test_optimization_levels(
                test_name=test_name,
                model_arch=model_arch,
                use_cache=use_cache,
                optimization_level=optimization_level,
                provider="CUDAExecutionProvider",
                use_io_binding=use_io_binding,
            )


class ORTOptimizerForCausalLMIntegrationTest(ORTOptimizerTestMixin):
    TASK = "causal-lm"
    ORTMODEL_CLASS = ORTModelForCausalLM

    SUPPORTED_ARCHITECTURES = [
        "bloom",
        "codegen",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "gptj",
    ]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_merged": [True, False],
    }

    def _test_optimization_levels(
        self,
        test_name: str,
        model_arch: str,
        use_cache: bool,
        use_merged: bool,
        optimization_level: str,
        provider: str,
        use_io_binding: Optional[bool] = None,
    ):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        export_name = test_name[:-3]  # remove `_OX` that is irrelevant as the export
        model_args = {
            "test_name": export_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        ort_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[export_name], use_cache=use_cache, provider=provider, use_io_binding=use_io_binding
        )

        if use_merged:
            with self.assertRaises(NotImplementedError) as cm:
                optimizer = ORTOptimizer.from_pretrained(ort_model)

            self.assertTrue("ORTModelForCausalLM models that use a single ONNX" in str(cm.exception))
            self.skipTest("Unsupported optimization case")
        else:
            optimizer = ORTOptimizer.from_pretrained(ort_model)

        if provider == "CUDAExecutionProvider":
            for_gpu = True
            device = "cuda"
        else:
            for_gpu = False
            device = "cpu"

        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level, for_gpu=for_gpu)
        optimization_config.disable_shape_inference = True
        model_id = MODEL_NAMES[model_arch]

        with tempfile.TemporaryDirectory(suffix="_optimized") as tmp_dir:
            optimizer.optimize(save_dir=tmp_dir, optimization_config=optimization_config)

            optimized_model = ORTModelForCausalLM.from_pretrained(
                tmp_dir, use_cache=use_cache, provider=provider, use_io_binding=use_io_binding
            )

            expected_ort_config = ORTConfig(optimization=optimization_config)
            ort_config = ORTConfig.from_pretrained(tmp_dir)

            # Verify the ORTConfig was correctly created and saved
            self.assertEqual(ort_config.to_dict(), expected_ort_config.to_dict())

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokens = tokenizer("This is a sample input", return_tensors="pt").to(device)

            model_outputs = ort_model.generate(**tokens)
            optimized_model_outputs = optimized_model.generate(**tokens)

            self.assertTrue(torch.equal(model_outputs, optimized_model_outputs))
            gc.collect()

    @parameterized.expand(
        grid_parameters({**FULL_GRID, "use_cache": [False, True], "optimization_level": ["O1", "O2", "O3"]})
    )
    def test_optimization_levels_cpu(
        self, test_name: str, model_arch: str, use_merged: bool, use_cache: bool, optimization_level: str
    ):
        self._test_optimization_levels(
            test_name=test_name,
            model_arch=model_arch,
            use_cache=use_cache,
            use_merged=use_merged,
            optimization_level=optimization_level,
            provider="CPUExecutionProvider",
        )

    @parameterized.expand(
        grid_parameters({**FULL_GRID, "use_cache": [True], "optimization_level": ["O1", "O2", "O3", "O4"]})
    )
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_optimization_levels_gpu(
        self, test_name: str, model_arch: str, use_merged: bool, use_cache: bool, optimization_level: str
    ):
        for use_io_binding in [False, True]:
            self._test_optimization_levels(
                test_name=test_name,
                model_arch=model_arch,
                use_cache=use_cache,
                use_merged=use_merged,
                optimization_level=optimization_level,
                provider="CUDAExecutionProvider",
                use_io_binding=use_io_binding,
            )
