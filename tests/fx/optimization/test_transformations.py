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
import operator
import unittest

import torch
from transformers import AutoTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.utils.fx import symbolic_trace

from optimum.fx.optimization import ChangeTrueDivToMulByInverse, MergeLinears, Transformation, compose
from optimum.fx.utils import are_fx_features_available
from parameterized import parameterized


_MODEL_NAME = "hf-internal-testing/tiny-random-bert"


class DummyTransformation(Transformation):
    def transform(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == operator.mul:
                node.target = operator.add
        return graph_module


_TRANSFORMATIONS_TO_TEST = ((DummyTransformation(),),)
_REVERSIBLE_TRANSFORMATIONS_TO_TEST = (
    (MergeLinears(),),
    (ChangeTrueDivToMulByInverse(),),
)


def get_bert_model():
    model = BertModel.from_pretrained(_MODEL_NAME)
    model.eval()
    traced = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
    return model, traced


class TransformationTester(unittest.TestCase):
    def flatten_output(self, output):
        flatten = []
        for x in output:
            if isinstance(x, (tuple, list)):
                flatten += self.flatten_output(x)
            elif not isinstance(x, torch.Tensor):
                continue
            else:
                flatten.append(x)
        return flatten

    def _check_original_and_transformed_outputs_match(self, transformation):
        model, traced = get_bert_model()
        transformed = transformation(traced)

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        inputs = tokenizer("This is a test.", return_tensors="pt")
        model_output = model(**inputs)
        transformed_output = transformed(**inputs)
        model_output = self.flatten_output(model_output)
        transformed_output = self.flatten_output(transformed_output)

        if transformation.preserves_computation:
            num_outputs = len(model_output)
            for i in range(num_outputs):
                self.assertTrue(
                    torch.allclose(model_output[i], transformed_output[i]),
                    f"transformed {i}th output doesn't match model {i}th output for {transformation.__class__.__name__}",
                )
        return inputs, model, transformed, model_output

    @parameterized.expand(_TRANSFORMATIONS_TO_TEST, skip_on_empty=True)
    @unittest.skipIf(not are_fx_features_available(), "not supported with this transformers version")
    def test_transformation(self, transformation):
        self._check_original_and_transformed_outputs_match(transformation)

    @parameterized.expand(_REVERSIBLE_TRANSFORMATIONS_TO_TEST)
    @unittest.skipIf(not are_fx_features_available(), "not supported with this transformers version")
    def test_reversible_transformation(self, transformation):
        inputs, model, transformed, model_output = self._check_original_and_transformed_outputs_match(transformation)

        restored = transformation(transformed, reverse=True)
        restored_output = restored(**inputs)
        restored_output = self.flatten_output(restored_output)

        num_outputs = len(model_output)
        for i in range(num_outputs):
            self.assertTrue(
                torch.allclose(model_output[i], restored_output[i]),
                f"restored {i}th output doesn't match model {i}th output for {transformation.__class__.__name__}",
            )

        orig_named_parameters = dict(model.named_parameters())
        restored_named_parameters = dict(restored.named_parameters())

        self.assertSetEqual(set(orig_named_parameters.keys()), set(restored_named_parameters.keys()))
        for name in orig_named_parameters:
            self.assertTrue(
                torch.allclose(orig_named_parameters[name], restored_named_parameters[name]),
                f"the {name} parameter does not match between the original and the restored models",
            )

    def _check_compose_works(self, inplace):
        _, traced = get_bert_model()
        transformed = MergeLinears()(ChangeTrueDivToMulByInverse()(traced))

        _, traced = get_bert_model()
        composition = compose(ChangeTrueDivToMulByInverse(), MergeLinears(), inplace=inplace)
        transformed_with_composition = composition(traced)

        self.assertEqual(transformed.code, transformed_with_composition.code)

        restored = MergeLinears()(ChangeTrueDivToMulByInverse()(transformed, reverse=True), reverse=True)
        restored_with_composition = composition(transformed_with_composition, reverse=True)

        self.assertEqual(restored.code, restored_with_composition.code)

    def test_compose_inplace(self):
        self._check_compose_works(True)

    def test_compose_deepcopy(self):
        self._check_compose_works(False)

    def test_compose_preserves(self):
        _, traced = get_bert_model()
        composition = compose(ChangeTrueDivToMulByInverse(), MergeLinears())
        self.assertTrue(composition.preserves_computation)

        composition = compose(DummyTransformation(), ChangeTrueDivToMulByInverse(), MergeLinears())
        self.assertFalse(composition.preserves_computation)


def test_merge_linears():
    _, traced = get_bert_model()
    transformation = MergeLinears()
    transformed = transformation(traced)

    self_attention_modules = [mod for mod in transformed.modules() if isinstance(mod, BertSelfAttention)]
    for module in self_attention_modules:
        num_linears = sum((1 if isinstance(mod, torch.nn.Linear) else 0 for mod in module.modules()))
        assert num_linears == 1, "there should be one linear layer in BertSelfAttention after merging"

    restored = transformation(transformed, reverse=True)
    self_attention_modules = [mod for mod in restored.modules() if isinstance(mod, BertSelfAttention)]
    for module in self_attention_modules:
        num_linears = sum((1 if isinstance(mod, torch.nn.Linear) else 0 for mod in module.modules()))
        assert (
            num_linears == 3
        ), "there should be three linear layers in BertSelfAttention after the reverse transformation"


def test_change_truediv_to_mul_by_inverse():
    _, traced = get_bert_model()

    orig_num_truediv = sum((1 if node.target == operator.truediv else 0 for node in traced.graph.nodes))
    orig_num_mul = sum((1 if node.target == operator.mul else 0 for node in traced.graph.nodes))

    transformation = ChangeTrueDivToMulByInverse()
    transformed = transformation(traced)

    num_truediv = sum((1 if node.target == operator.truediv else 0 for node in transformed.graph.nodes))
    num_mul = sum((1 if node.target == operator.mul else 0 for node in transformed.graph.nodes))

    assert (orig_num_truediv != num_truediv) and (orig_num_mul != num_mul)
    assert orig_num_truediv - num_truediv == num_mul - orig_num_mul

    restored = transformation(traced, reverse=True)

    restored_num_truediv = sum((1 if node.target == operator.truediv else 0 for node in restored.graph.nodes))
    restored_num_mul = sum((1 if node.target == operator.mul else 0 for node in restored.graph.nodes))

    assert (orig_num_truediv == restored_num_truediv) and (orig_num_mul == restored_num_mul)
