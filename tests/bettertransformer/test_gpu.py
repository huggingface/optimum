import os
import unittest

import torch
from parameterized import parameterized
from transformers import AutoModel

from optimum.bettertransformer import BetterTransformer
from optimum.utils import logging
from optimum.utils.testing_utils import grid_parameters


logger = logging.get_logger()
logging.set_verbosity_info()


def timing_cuda(model, num_batches, input_ids, masks, decoder_input_ids):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_batches):
        _ = model(input_ids, masks, decoder_input_ids=decoder_input_ids)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches


def benchmark(model_name: str, num_batches: int, batch_size: int, max_seqlen: int, is_half: bool):
    hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if is_half else None).eval()
    hf_model = hf_model.to("cuda:0")
    bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

    vocab_size = hf_model.config.vocab_size
    input_ids = torch.randint(vocab_size - 1, (batch_size, max_seqlen), dtype=torch.int64) + 1
    masks = torch.ones(batch_size, max_seqlen, dtype=torch.int64)

    input_ids = input_ids.to("cuda:0")
    masks = masks.to("cuda:0")

    decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.int64).to("cuda:0")

    # Warmup
    _ = hf_model(input_ids, masks, decoder_input_ids=decoder_input_ids)
    torch.cuda.synchronize()
    _ = bt_model(input_ids, masks, decoder_input_ids=decoder_input_ids)
    torch.cuda.synchronize()

    total_hf_time = timing_cuda(hf_model, num_batches, input_ids, masks, decoder_input_ids)
    total_bt_time = timing_cuda(bt_model, num_batches, input_ids, masks, decoder_input_ids)

    return total_bt_time, total_hf_time


class TestSpeedup(unittest.TestCase):
    """
    TODO: test missing for:

    - WhisperEncoderLayerBetterTransformer
    - ViTLayerBetterTransformer
    - ViltLayerBetterTransformer
    - Wav2Vec2EncoderLayerBetterTransformer
    - FSMTEncoderLayerBetterTransformer
    - CLIPLayerBetterTransformer
    """

    REPRESENTATIVE_MODELS = [
        "bert-base-uncased",
        # "albert-base-v2",  # TODO: AlbertLayerBetterTransformer seem to nest/unnest tensors all the time
        "facebook/bart-base",
        "facebook/mbart-large-50",
        "distilbert-base-uncased",
    ]

    @parameterized.expand(
        grid_parameters(
            {
                "model_name": REPRESENTATIVE_MODELS,
                "batch_size": [32, 64],
                "sequence_length": [64, 128],
                "use_half": [True, False],
            }
        )
    )
    @unittest.skipIf(int(os.environ.get("TEST_LEVEL", 0)) < 1, reason="disabled by default")
    def test_base_speedup(
        self, test_name: str, model_name: str, batch_size: int, sequence_length: int, use_half: bool
    ):
        """
        Test to validate the BetterTransformer base speedup on GPU.

        The speedup check is low because we still hit https://github.com/pytorch/pytorch/issues/91305
        """
        num_batches = 50

        total_bt_time, total_hf_time = benchmark(
            model_name,
            num_batches,
            batch_size,
            sequence_length,
            use_half,
        )

        speedup = total_hf_time / total_bt_time

        self.assertTrue(speedup > 0.85, msg=f"The BetterTransformer base speedup for {test_name} is {speedup}")

        if speedup >= 0.85 and speedup < 1:
            logger.warning(f"The BetterTransformer base speedup for {test_name} is {speedup}")
        if speedup >= 1:
            logger.info(f"The BetterTransformer base speedup for {test_name} is {speedup}")
