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
from parameterized import parameterized
from transformers import AutoModel, AutoModelForImageClassification, AutoTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.utils.fx import symbolic_trace

from optimum.fx.optimization import (
    ChangeTrueDivToMulByInverse,
    FuseBatchNorm1dInLinear,
    FuseBatchNorm2dInConv2d,
    FuseBiasInLinear,
    MergeLinears,
    ReversibleTransformation,
    Transformation,
    compose,
)
from optimum.fx.utils import are_fx_features_available


_MODEL_NAME = "hf-internal-testing/tiny-random-bert"


class DummyTransformation(Transformation):
    def __init__(self, some_argument=None):
        self.some_argument = some_argument

    def transform(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == operator.mul:
                node.target = operator.add
        return graph_module


_TRANSFORMATIONS_TO_TEST = (
    (DummyTransformation(), BertModel, _MODEL_NAME, ["input_ids", "attention_mask", "token_type_ids"], None, False),
    (
        FuseBatchNorm2dInConv2d(),
        AutoModelForImageClassification,
        "fxmarty/resnet-tiny-beans",
        ["pixel_values"],
        {"pixel_values": torch.rand(8, 3, 224, 224)},
        False,
    ),
    (
        FuseBatchNorm1dInLinear(),
        AutoModel,
        "hf-internal-testing/tiny-random-groupvit",
        ["input_ids", "attention_mask", "pixel_values"],
        {
            "pixel_values": torch.rand(2, 3, 30, 30, dtype=torch.float32),
            "input_ids": torch.randint(low=0, high=100, size=(2, 32)),
            "attention_mask": torch.ones(2, 32),
        },
        True,
    ),
)
_REVERSIBLE_TRANSFORMATIONS_TO_TEST = (
    (MergeLinears(), BertModel, _MODEL_NAME, ["input_ids", "attention_mask", "token_type_ids"], None, False),
    (FuseBiasInLinear(), BertModel, _MODEL_NAME, ["input_ids", "attention_mask", "token_type_ids"], None, False),
    (
        ChangeTrueDivToMulByInverse(),
        BertModel,
        _MODEL_NAME,
        ["input_ids", "attention_mask", "token_type_ids"],
        None,
        False,
    ),
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

    def _check_original_and_transformed_outputs_match(
        self, transformation, model_class, model_name, model_input_names, inputs, disable_check
    ):
        model = model_class.from_pretrained(model_name)
        model.eval()
        traced = symbolic_trace(model, input_names=model_input_names, disable_check=disable_check)

        if inputs is None:
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
            inputs = tokenizer("This is a test.", return_tensors="pt")

        model_output = model(**inputs)
        transformed = transformation(traced)
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
    def test_transformation(
        self, transformation, model_class, model_name, model_input_names, dummy_inputs, disable_check
    ):
        self._check_original_and_transformed_outputs_match(
            transformation, model_class, model_name, model_input_names, dummy_inputs, disable_check
        )

    @parameterized.expand(_REVERSIBLE_TRANSFORMATIONS_TO_TEST)
    @unittest.skipIf(not are_fx_features_available(), "not supported with this transformers version")
    def test_reversible_transformation(
        self, transformation, model_class, model_name, model_input_names, dummy_inputs, disable_check
    ):
        inputs, model, transformed, model_output = self._check_original_and_transformed_outputs_match(
            transformation, model_class, model_name, model_input_names, dummy_inputs, disable_check
        )

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

    def test_transformation_signature(self):
        t1 = DummyTransformation()
        t2 = DummyTransformation(some_argument=1)

        # Same transformation
        self.assertEqual(t1.signature, DummyTransformation().signature)
        # Same transformation class, but different attributes
        self.assertNotEqual(t1.signature, t2.signature)
        # Different transformation class, but same attributes
        DifferentTransformation = type("DifferentTransformation", (DummyTransformation,), {})
        t3 = DifferentTransformation(some_argument=1)
        self.assertNotEqual(t2.signature, t3.signature)

    def test_transformation_mark_and_transformed(self):
        _, traced = get_bert_model()

        class MarkLinears(ReversibleTransformation):
            def __init__(self, some_argument=None):
                self.some_argument = some_argument

            def transform(self, graph_module):
                for node in graph_module.graph.nodes:
                    if node.op == "call_module" and isinstance(
                        graph_module.get_submodule(node.target), torch.nn.Linear
                    ):
                        self.mark_as_transformed(node)
                return graph_module

            def reverse(self, graph_module):
                for node in graph_module.graph.nodes:
                    if node.op == "call_module" and isinstance(
                        graph_module.get_submodule(node.target), torch.nn.Linear
                    ):
                        self.mark_as_restored(node)
                return graph_module

        # Marked by one transform.
        t1 = MarkLinears()
        traced = t1(traced)
        for node in traced.graph.nodes:
            if node.op == "call_module" and isinstance(traced.get_submodule(node.target), torch.nn.Linear):
                self.assertTrue(t1.transformed(node))
        # Not marked by another transform.
        t2 = MarkLinears(some_argument="test")
        for node in traced.graph.nodes:
            if node.op == "call_module" and isinstance(traced.get_submodule(node.target), torch.nn.Linear):
                self.assertFalse(t2.transformed(node))
        # And now marked by it.
        traced = t2(traced)
        for node in traced.graph.nodes:
            if node.op == "call_module" and isinstance(traced.get_submodule(node.target), torch.nn.Linear):
                self.assertTrue(t2.transformed(node))
        # Reversed first transform, nodes should not be marked as transformed anymore.
        traced = t1(traced, reverse=True)
        for node in traced.graph.nodes:
            if node.op == "call_module" and isinstance(traced.get_submodule(node.target), torch.nn.Linear):
                self.assertFalse(t1.transformed(node))


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


def test_fuse_bias_in_linear():
    _, traced = get_bert_model()
    num_bias_in_linears = sum(
        int(mod.bias is not None) for mod in traced.modules() if isinstance(mod, torch.nn.Linear)
    )
    assert (
        num_bias_in_linears != 0
    ), "there should be biases in at least one linear module in the model to actually perform the test"
    transformation = FuseBiasInLinear()
    transformed = transformation(traced)
    num_bias_in_linears = sum(
        int(mod.bias is not None) for mod in transformed.modules() if isinstance(mod, torch.nn.Linear)
    )
    assert num_bias_in_linears == 0, "there should not be any bias left in any linear module now"


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


def test_fuse_conv2d_batchnorm2d():
    model = AutoModelForImageClassification.from_pretrained("fxmarty/resnet-tiny-beans")

    traced_model = symbolic_trace(model, input_names=["pixel_values"], disable_check=True)

    num_batchnorm2d = sum(1 if isinstance(mod, torch.nn.BatchNorm2d) else 0 for mod in traced_model.modules())
    assert num_batchnorm2d != 0, "there should be at least one BatchNorm2d in the model to actually perform the test"

    transformation = FuseBatchNorm2dInConv2d()
    transformed_model = transformation(traced_model)

    num_batchnorm2d = sum(1 if isinstance(mod, torch.nn.BatchNorm2d) else 0 for mod in transformed_model.modules())
    assert num_batchnorm2d == 0, "there should be no BatchNorm2d left in the model after the transformation"


def test_fuse_linear_batchnorm1d():
    model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-groupvit")
    model.eval()

    traced_model = symbolic_trace(
        model, input_names=["input_ids", "attention_mask", "pixel_values"], disable_check=True
    )

    num_batchnorm1d = sum(1 if isinstance(mod, torch.nn.BatchNorm1d) else 0 for mod in traced_model.modules())
    assert num_batchnorm1d != 0, "there should be at least one BatchNorm1d in the model to actually perform the test"

    transformation = FuseBatchNorm1dInLinear()
    transformed_model = transformation(traced_model)

    num_batchnorm1d = sum(1 if isinstance(mod, torch.nn.BatchNorm1d) else 0 for mod in transformed_model.modules())
    assert num_batchnorm1d == 0, "there should be no BatchNorm1d left in the model after the transformation"


def test_get_parent():
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(10, 20, kernel_size=3)
            self.bn2d = torch.nn.BatchNorm2d(20)

        def forward(self, x):
            x = self.conv2d(x)
            x = self.bn2d(x)
            return x

    model = MyModel()
    model.eval()

    traced_model = torch.fx.symbolic_trace(model)

    for node in traced_model.graph.nodes:
        if node.target == "conv2d":
            parent_name, _, name = node.target.rpartition(".")
            assert parent_name == ""
            parent_module = traced_model.get_submodule(parent_name)
            assert parent_module == traced_model
