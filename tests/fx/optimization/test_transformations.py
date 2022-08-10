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
from transformers import AutoModelForImageClassification, AutoTokenizer, BertModel, GroupViTConfig, GroupViTModel
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
from parameterized import parameterized


_MODEL_NAME = "hf-internal-testing/tiny-random-bert"


class DummyTransformation(Transformation):
    def __init__(self, some_argument=None):
        self.some_argument = some_argument

    def transform(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == operator.mul:
                node.target = operator.add
        return graph_module


_TRANSFORMATIONS_TO_TEST = ((DummyTransformation(),),)
_REVERSIBLE_TRANSFORMATIONS_TO_TEST = (
    (MergeLinears(),),
    (FuseBiasInLinear(),),
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


class CustomTransformationsTests(unittest.TestCase):
    def test_fuse_bias_in_linear(self):
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

    def test_merge_linears(self):
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

    def test_change_truediv_to_mul_by_inverse(self):
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

    def test_fuse_conv2d_batchnorm2d(self):
        model = AutoModelForImageClassification.from_pretrained("fxmarty/resnet-tiny-beans")

        traced_model = symbolic_trace(model, input_names=["pixel_values"], disable_check=True)

        num_batchnorm2d = sum(1 if isinstance(mod, torch.nn.BatchNorm2d) else 0 for mod in traced_model.modules())
        self.assertNotEqual(
            num_batchnorm2d,
            0,
            msg="there should be at least one BatchNorm2d in the model to actually perform the test",
        )

        transformation = FuseBatchNorm2dInConv2d()
        transformed_model = transformation(traced_model)

        num_batchnorm2d = sum(1 if isinstance(mod, torch.nn.BatchNorm2d) else 0 for mod in transformed_model.modules())
        self.assertEqual(
            num_batchnorm2d, 0, msg="there should be no BatchNorm2d left in the model after the transformation"
        )

        dummy_input = torch.rand(8, 3, 224, 224)

        output_original = model(pixel_values=dummy_input)
        output_transformed = transformed_model(pixel_values=dummy_input)

        self.assertTrue(torch.allclose(output_original.logits, output_transformed["logits"], atol=1e-6))

    def test_fuse_linear_batchnorm1d(self):
        config = GroupViTConfig()
        model = GroupViTModel(config)
        model.text_projection[1].weight.data = torch.rand(model.text_projection[1].weight.data.shape)
        model.text_projection[1].bias.data = torch.rand(model.text_projection[1].bias.data.shape)
        model.eval()

        traced_model = symbolic_trace(
            model, input_names=["input_ids", "attention_mask", "pixel_values"], disable_check=True
        )

        num_batchnorm1d = sum(1 if isinstance(mod, torch.nn.BatchNorm1d) else 0 for mod in traced_model.modules())
        self.assertNotEqual(
            num_batchnorm1d,
            0,
            msg="there should be at least one BatchNorm1d in the model to actually perform the test",
        )

        transformation = FuseBatchNorm1dInLinear()
        transformed_model = transformation(traced_model)

        num_batchnorm1d = sum(1 if isinstance(mod, torch.nn.BatchNorm1d) else 0 for mod in transformed_model.modules())
        self.assertEqual(
            num_batchnorm1d, 0, msg="there should be no BatchNorm1d left in the model after the transformation"
        )

        dummy_input = {
            "pixel_values": torch.rand(1, 3, 224, 224, dtype=torch.float32),
            "input_ids": torch.randint(low=0, high=100, size=(2, 32)),
            "attention_mask": torch.ones(2, 32),
        }

        output_original = model(**dummy_input)
        output_transformed = transformed_model(**dummy_input)

        self.assertTrue(torch.allclose(output_transformed["logits_per_image"], output_original.logits_per_image))
        self.assertTrue(torch.allclose(output_transformed["logits_per_text"], output_original.logits_per_text))
        self.assertTrue(torch.allclose(output_transformed["text_embeds"], output_original.text_embeds, atol=1e-6))
        self.assertTrue(torch.allclose(output_transformed["image_embeds"], output_original.image_embeds, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                output_transformed["text_model_output"]["last_hidden_state"],
                output_original.text_model_output.last_hidden_state,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output_transformed["text_model_output"]["pooler_output"],
                output_original.text_model_output.pooler_output,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output_transformed["vision_model_output"]["last_hidden_state"],
                output_original.vision_model_output.last_hidden_state,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output_transformed["vision_model_output"]["pooler_output"],
                output_original.vision_model_output.pooler_output,
                atol=1e-6,
            )
        )
