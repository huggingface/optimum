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

import copy
import inspect
import tempfile
import unittest

import torch
from transformers import AutoModel

from optimum.bettertransformer import BetterTransformer
from optimum.utils.testing_utils import flatten_dict, require_torch_gpu


MODELS_DICT = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "bark": "ylacombe/bark-small",
    "bart": "hf-internal-testing/tiny-random-bart",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "bert-generation": "ybelkada/random-tiny-BertGenerationModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blip-2": "hf-internal-testing/tiny-random-Blip2Model",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "clip_text_model": "hf-internal-testing/tiny-random-clip-zero-shot-image-classification",  # with quick_gelu
    "clip": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",  # with gelu
    "codegen": "hf-internal-testing/tiny-random-CodeGenModel",
    "data2vec-text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "deit": "hf-internal-testing/tiny-random-deit",
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "ernie": "hf-internal-testing/tiny-random-ErnieModel",
    # NOTE: falcon directly supports SDPA in Transformers.
    # "falcon": "fxmarty/really-tiny-falcon-testing",
    "fsmt": "hf-internal-testing/tiny-random-FSMTModel",
    "gpt2": "hf-internal-testing/tiny-random-GPT2Model",
    # NOTE: this tiny model does not use attention_softmax_in_fp32=True (contrary to e.g. starcoder)
    # NOTE: gpt_bigcode directy supports SDPA in Transformers.
    # "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "hubert": "ybelkada/hubert-tiny-random",
    "layoutlm": "hf-internal-testing/tiny-random-LayoutLMModel",
    # NOTE: llama directy supports SDPA in Transformers.
    # "llama": "fxmarty/tiny-llama-fast-tokenizer",
    # "llama-gqa": "noamwies/llama-test-gqa-with-better-transformer",
    "m2m_100": "hf-internal-testing/tiny-random-nllb",
    "marian": "optimum-internal-testing/tiny-random-marian",  # the other tiny ones have a too small max_position_embeddings
    "markuplm": "hf-internal-testing/tiny-random-MarkupLMModel",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "pegasus": "hf-internal-testing/tiny-random-PegasusModel",
    "prophetnet": "optimum-internal-testing/tiny-random-prophetnet",  # the other tiny ones have a too small max_position_embeddings
    "rembert": "hf-internal-testing/tiny-random-RemBertModel",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "rocbert": "hf-internal-testing/tiny-random-RoCBertModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "splinter": "hf-internal-testing/tiny-random-SplinterModel",
    "tapas": "hf-internal-testing/tiny-random-TapasModel",
    "t5": "hf-internal-testing/tiny-random-t5",
    "vilt": "hf-internal-testing/tiny-vilt-random-vqa",
    "vit": "hf-internal-testing/tiny-random-ViTModel",
    "vit_mae": "hf-internal-testing/tiny-random-ViTMAEModel",
    "vit_msn": "hf-internal-testing/tiny-random-ViTMSNModel",
    "wav2vec2": ("patrickvonplaten/wav2vec2_tiny_random", "ybelkada/tiny-wav2vec2-stable-ln"),
    # NOTE: whisper directy supports SDPA in Transformers.
    # "whisper": "openai/whisper-tiny",
    "xlm_roberta": "hf-internal-testing/tiny-xlm-roberta",
    "yolos": "hf-internal-testing/tiny-random-YolosModel",
}

known_dropout_keys = [
    "attention_probs_dropout_prob",
    "hidden_dropout_prob",
    "classifier_dropout_prob",
    "attention_dropout",
    "hidden_dropout",
    "dropout",
    "qa_dropout",
    "seq_classif_dropout",
    "summary_last_dropout",
    "classifier_dropout",
    "activation_dropout",
    "classif_dropout",
    "dropout_rate",
    "attn_pdrop",
    "embd_pdrop",
    "resid_pdrop",
    "summary_first_dropout",
]


def set_dropout_to_zero(config):
    for attr_name in known_dropout_keys:
        if hasattr(config, attr_name):
            setattr(config, attr_name, 0.0)

    return config


class BetterTransformersTestMixin(unittest.TestCase):
    r"""
    `BetterTransformersTestMixin` to wrap necessary functions for testing `BetterTransformer`
    integration. This includes the following tests:
        - `test_logits`: This tests if the converted model produces the same logits
        than the original model.
        - `test_raise_on_save`: Test if the converion properly raises an error if someone tries to save the model using `save_pretrained`.
    """

    def prepare_inputs_for_class(self, model_id=None, model_type=None):
        raise NotImplementedError

    @require_torch_gpu
    def _test_fp16_inference(
        self, model_id: str, model_type: str, automodel_class, use_to_operator=False, **preprocessor_kwargs
    ):
        r"""
        This tests if the converted model runs fine under fp16.
        """
        # The first row of the attention mask needs to be all ones -> check: https://github.com/pytorch/pytorch/blob/19171a21ee8a9cc1a811ac46d3abd975f0b6fc3b/test/test_nn.py#L5283
        inputs = self.prepare_inputs_for_class(model_id=model_id, model_type=model_type, **preprocessor_kwargs).to(0)

        torch.manual_seed(0)
        if not use_to_operator:
            hf_random_model = automodel_class.from_pretrained(model_id, torch_dtype=torch.float16).to(0)
            converted_model = BetterTransformer.transform(hf_random_model, keep_original_model=True)
        else:
            hf_random_model = automodel_class.from_pretrained(model_id).to(0)
            converted_model = BetterTransformer.transform(hf_random_model, keep_original_model=True)
            hf_random_model = hf_random_model.to(torch.float16)
            converted_model = converted_model.to(torch.float16)

        self.assertFalse(
            hasattr(hf_random_model, "use_bettertransformer"),
            f"The model {hf_random_model.__class__.__name__} has been converted to a `fast` model by mistake.",
        )

        length = 50
        with torch.no_grad():
            r"""
            Make sure the models are in eval mode! Make also sure that the original model
            has not been converted to a fast model. The check is done above.
            """
            torch.manual_seed(0)
            output_hf = hf_random_model.generate(**inputs, min_length=length, max_length=length)

            torch.manual_seed(0)
            output_bt = converted_model.generate(**inputs, min_length=length, max_length=length)

            self.assertTrue(
                torch.allclose(output_hf, output_bt),
                f"Maxdiff: {(output_hf - output_bt).abs().max()}",
            )

    def _test_logits_backward(self, model_id: str, model_type: str, **preprocessor_kwargs):
        inputs = self.prepare_inputs_for_class(model_id=model_id, model_type=model_type, **preprocessor_kwargs)

        hf_random_model = AutoModel.from_pretrained(model_id).eval()
        random_config = hf_random_model.config

        # I could not obtain reproducible results with `torch.manual_seed` nor with
        # `torch.random.set_rng_state`. An alternative could be to make dropout stateful,
        # and to replace them with a static pattern for this test. Currently, we use
        # functional dropout though.
        # We need to be in train mode to take the right path.
        random_config = set_dropout_to_zero(random_config)

        # m2m_100 randomly drops layers, which makes testing flaky (see `skip_the_layer` in transformers, some other models use it as well)
        if model_type == "m2m_100":
            random_config.encoder_layerdrop = 0
            random_config.decoder_layerdrop = 0

        hf_random_model = hf_random_model.__class__(random_config)

        converted_model = copy.deepcopy(hf_random_model)
        converted_model = BetterTransformer.transform(converted_model)

        hf_random_model = hf_random_model.train()
        converted_model = converted_model.train()

        optimizer_hf = torch.optim.SGD(hf_random_model.parameters(), lr=0.2)
        optimizer_bt = torch.optim.SGD(converted_model.parameters(), lr=0.2)

        tol = 2e-3

        hf_hidden_states = hf_random_model(**inputs)[0]
        bt_hidden_states = converted_model(**inputs)[0]

        self.assert_equal(
            hf_hidden_states,
            bt_hidden_states,
            atol=tol,
            model_name=hf_random_model.__class__.__name__,
        )

        loss_hf = hf_hidden_states.abs().mean()
        loss_bt = bt_hidden_states.abs().mean()

        loss_hf.backward()
        loss_bt.backward()

        optimizer_hf.step()
        optimizer_bt.step()

        hf_hidden_states = hf_random_model(**inputs)[0]
        bt_hidden_states = converted_model(**inputs)[0]

        self.assert_equal(
            hf_hidden_states,
            bt_hidden_states,
            atol=tol,
            model_name=hf_random_model.__class__.__name__,
        )

    def _test_logits(self, model_id: str, model_type: str, **preprocessor_kwargs):
        r"""
        This tests if the converted model produces the same logits
        as the original model.
        """
        # The first row of the attention mask needs to be all ones -> check: https://github.com/pytorch/pytorch/blob/19171a21ee8a9cc1a811ac46d3abd975f0b6fc3b/test/test_nn.py#L5283
        inputs = self.prepare_inputs_for_class(model_id=model_id, model_type=model_type, **preprocessor_kwargs)

        torch.manual_seed(0)
        hf_random_model = AutoModel.from_pretrained(model_id, attn_implementation="eager").eval()
        random_config = hf_random_model.config

        hf_random_model = hf_random_model.eval()

        torch.manual_seed(0)
        converted_model = BetterTransformer.transform(hf_random_model, keep_original_model=True)

        self.assertFalse(hf_random_model.training)
        self.assertFalse(converted_model.training)
        self.assertFalse(
            hasattr(hf_random_model, "use_bettertransformer"),
            f"The model {hf_random_model.__class__.__name__} has been converted to a `fast` model by mistake.",
        )

        with torch.no_grad():
            r"""
            Make sure the models are in eval mode! Make also sure that the original model
            has not been converted to a fast model. The check is done above.
            """
            torch.manual_seed(0)
            hf_hidden_states = hf_random_model(**inputs)[0]

            torch.manual_seed(0)
            bt_hidden_states = converted_model(**inputs)[0]

            if "quick_gelu" in flatten_dict(random_config.to_dict()).values():
                # Since `quick_gelu` is a rather slightly modified version of `GeLU` we expect a discrepency.
                tol = 3e-1
            elif "gelu_new" in flatten_dict(random_config.to_dict()).values():
                # Since `gelu_new` is a slightly modified version of `GeLU` we expect a small
                # discrepency.
                tol = 4e-2
            else:
                tol = 2e-3

            if hasattr(self, "compare_outputs"):
                self.compare_outputs(
                    model_type,
                    hf_hidden_states,
                    bt_hidden_states,
                    atol=tol,
                    model_name=hf_random_model.__class__.__name__,
                )
            elif "attention_mask" in inputs:
                for i, attention_mask in enumerate(inputs["attention_mask"]):
                    length = torch.argwhere(attention_mask != 0).max().item()
                    self.assert_equal(
                        tensor1=hf_hidden_states[i, : length + 1, :],
                        tensor2=bt_hidden_states[i, : length + 1, :],
                        atol=tol,
                        model_name=hf_random_model.__class__.__name__,
                    )
            else:
                self.assert_equal(
                    tensor1=hf_hidden_states[:, :3, :],
                    tensor2=bt_hidden_states[:, :3, :],
                    atol=tol,
                    model_name=hf_random_model.__class__.__name__,
                )

    def assert_equal(self, tensor1, tensor2, atol: float, model_name: str):
        self.assertTrue(
            torch.allclose(tensor1, tensor2, atol=atol),
            f"The BetterTransformer converted model does not produce the same logits as the original model. Failed for the model {model_name}."
            f" Maxdiff: {torch.abs(tensor1 - tensor2).max()}",
        )

    def _test_train_decoder(self, model_id: str, model_type: str, **kwargs):
        r"""
        A tests that checks if the training works as expected for decoder models.
        """
        inputs = self.prepare_inputs_for_class(model_id=model_id, model_type=model_type, **kwargs)

        hf_random_model = AutoModel.from_pretrained(model_id).eval()

        bt_model = BetterTransformer.transform(hf_random_model, keep_original_model=True)
        bt_model.train()
        output = bt_model(**inputs)[0]
        loss = output.sum()
        loss.backward()

        # check if gradients are not None
        for name, param in bt_model.named_parameters():
            # TODO: is this normal? None even without bettertransformer
            if hf_random_model.config.model_type == "pegasus" and "embed_positions.weight" in name:
                continue
            self.assertIsNotNone(param.grad)

    def _test_invert_modules(self, model_id, keep_original_model=False):
        r"""
        Test that the inverse converted model and hf model have the same modules
        """
        hf_model = AutoModel.from_pretrained(model_id)
        hf_modules = list(hf_model.modules())

        bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)
        bt_model = BetterTransformer.reverse(bt_model)

        bt_modules = list(bt_model.modules())

        self.assertEqual(len(hf_modules), len(bt_modules))
        for hf_module, bt_module in zip(hf_modules, bt_modules):
            # check the modules have the same signature and code
            # for the `forward` and `__init__` methods
            # as those are the only functions we change
            self.assertEqual(inspect.signature(hf_module.forward), inspect.signature(bt_module.forward))
            self.assertEqual(inspect.signature(hf_module.__init__), inspect.signature(bt_module.__init__))

            self.assertEqual(inspect.getsource(hf_module.forward), inspect.getsource(bt_module.forward))
            self.assertEqual(inspect.getsource(hf_module.__init__), inspect.getsource(bt_module.__init__))

    def _test_save_load_invertible(self, model_id, keep_original_model=True):
        with tempfile.TemporaryDirectory() as tmpdirname:
            hf_model = AutoModel.from_pretrained(model_id).eval()
            hf_model_state_dict = copy.deepcopy(hf_model.state_dict())

            bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)

            bt_model = BetterTransformer.reverse(bt_model)

            for name, param in bt_model.named_parameters():
                self.assertFalse(param.device.type == "meta", f"Parameter {name} is on the meta device.")

            # saving a normal transformers bark model fails because of shared tensors
            bt_model.save_pretrained(tmpdirname, safe_serialization=hf_model.config.model_type != "bark")

            bt_model_from_load = AutoModel.from_pretrained(tmpdirname)

            self.assertEqual(
                set(bt_model.state_dict().keys()),
                set(bt_model_from_load.state_dict().keys()),
            )

            self.assertEqual(
                hf_model_state_dict.keys(),
                set(bt_model_from_load.state_dict().keys()),
            )

            for key in bt_model.state_dict().keys():
                self.assertTrue(
                    torch.allclose(
                        bt_model.state_dict()[key],
                        bt_model_from_load.state_dict()[key],
                    )
                )

                self.assertTrue(
                    torch.allclose(
                        hf_model_state_dict[key],
                        bt_model_from_load.state_dict()[key],
                    )
                )

    def _test_invert_model_logits(
        self, model_id: str, model_type: str, keep_original_model=True, **preprocessor_kwargs
    ):
        r"""
        Test that the inverse converted model and hf model have the same logits
        """
        inputs = self.prepare_inputs_for_class(model_id, model_type=model_type, **preprocessor_kwargs)

        hf_model = AutoModel.from_pretrained(model_id)
        hf_model = hf_model.eval()

        with torch.inference_mode():
            torch.manual_seed(42)
            output_hf = hf_model(**inputs)

            bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)
            bt_model = BetterTransformer.reverse(bt_model)

            torch.manual_seed(42)
            output_bt = bt_model(**inputs)

        for i in range(len(output_bt)):
            if isinstance(output_bt[i], torch.Tensor):
                self.assertTrue(
                    torch.allclose(output_bt[i], output_hf[i], atol=1e-4),
                    f" Maxdiff: {(output_bt[i] - output_hf[i]).abs().max()}",
                )
            elif isinstance(output_bt[i], tuple):
                flattened_output_bt = [out for j in range(len(output_bt[i])) for out in output_bt[i][j]]
                flattened_output_hf = [out for j in range(len(output_hf[i])) for out in output_hf[i][j]]
                for j in range(len(flattened_output_bt)):
                    if isinstance(flattened_output_bt[j], torch.Tensor):
                        self.assertTrue(
                            torch.allclose(flattened_output_bt[j], flattened_output_hf[j], atol=1e-4),
                            f" Maxdiff: {(flattened_output_bt[j] - flattened_output_hf[j]).abs().max()}",
                        )
                    elif isinstance(flattened_output_bt[j], tuple):
                        for k in range(len(flattened_output_bt[j])):
                            self.assertTrue(
                                torch.allclose(flattened_output_bt[j][k], flattened_output_hf[j][k], atol=1e-4),
                                f" Maxdiff: {(flattened_output_bt[j][k] - flattened_output_hf[j][k]).abs().max()}",
                            )


def get_batch(batch_size, avg_seqlen, max_sequence_length, seqlen_stdev, vocab_size, pad_idx=0):
    r"""
    Utility function to generate a batch of random sequences, together with their
    attention mask and lengths.
    Copied from: https://github.com/HamidShojanazeri/transformers/blob/ddf0299a13e7c4f54459a0731abd80204a1078f5/examples/pytorch/benchmarking/benchmark_bettertransformer.py#L149
    """
    mean_tensor = torch.Tensor([avg_seqlen]).expand(batch_size)
    stdev_tensor = torch.Tensor([seqlen_stdev]).expand(batch_size)
    lengths = torch.normal(mean_tensor, stdev_tensor).to(torch.int)

    # need at least a sequence length of 1 for BetterTransformer to work
    lengths = torch.clamp(lengths, min=1, max=max_sequence_length)

    tokens = torch.full(
        (batch_size, max_sequence_length),
        pad_idx,
    )
    for i in range(batch_size):
        tokens[i, : lengths[i]] = torch.randint(
            pad_idx + 1,
            vocab_size - 1,
            size=(lengths[i],),
        )
    mask = torch.full(
        (batch_size, max_sequence_length),
        0,
    )
    for i in range(batch_size):
        mask[i, : lengths[i]] = 1
    return tokens, lengths, mask
