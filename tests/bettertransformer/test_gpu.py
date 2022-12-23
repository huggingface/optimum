import unittest

import torch
from transformers import AutoModel

from optimum.bettertransformer import BetterTransformer
from optimum.utils.testing_utils import grid_parameters
from parameterized import parameterized


def timing_cuda(model, num_batches, input_ids, masks):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_batches):
        _ = model(input_ids, masks)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches


def benchmark(model_name, num_batches, batch_size, max_seqlen, is_half):
    print("Loading model {}".format(model_name))
    hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if is_half else None).eval()
    hf_model = hf_model.to("cuda:0")
    bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

    vocab_size = 30522
    input_ids = torch.randint(vocab_size - 1, (batch_size, max_seqlen), dtype=torch.int64) + 1
    masks = torch.ones(batch_size, max_seqlen, dtype=torch.int64)

    input_ids = input_ids.to("cuda:0")
    masks = masks.to("cuda:0")

    # Warmup
    _ = hf_model(input_ids[0].unsqueeze(0), masks[0].unsqueeze(0))
    torch.cuda.synchronize()
    _ = bt_model(input_ids[0].unsqueeze(0), masks[0].unsqueeze(0))
    torch.cuda.synchronize()

    total_hf_time = timing_cuda(hf_model, num_batches, input_ids, masks)
    total_bt_time = timing_cuda(bt_model, num_batches, input_ids, masks)

    return total_bt_time, total_hf_time


class TestSpeedup(unittest.TestCase):
    @parameterized.expand(
        grid_parameters(
            {
                "model_name": ["bert-base-uncased"],
                "batch_size": [32, 64],
                "sequence_length": [64, 128, 256],
                "use_half": [True, False],
            }
        )
    )
    def test_base_speedup(
        self, test_name: str, model_name: str, batch_size: int, sequence_length: int, use_half: bool
    ):
        num_batches = 50

        total_bt_time, total_hf_time = benchmark(
            model_name,
            num_batches,
            batch_size,
            sequence_length,
            use_half,
        )

        speedup = total_hf_time / total_bt_time

        self.assertTrue(speedup > 1, msg="The BetterTransformer base speedup is < 1")
