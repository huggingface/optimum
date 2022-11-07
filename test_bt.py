import timeit

import torch
from transformers import AutoModel

from optimum.bettertransformer import BetterTransformer


def get_batch(batch_size, avg_seqlen, max_sequence_length, seqlen_stdev, vocab_size, pad_idx=0):
    r"""
    Utility function to generate a batch of random sequences, together with their
    attention mask and lengths.
    Copied from: https://github.com/HamidShojanazeri/transformers/blob/ddf0299a13e7c4f54459a0731abd80204a1078f5/examples/pytorch/benchmarking/benchmark_bettertransformer.py#L149
    """
    mean_tensor = torch.Tensor([avg_seqlen]).expand(batch_size)
    stdev_tensor = torch.Tensor([seqlen_stdev]).expand(batch_size)
    lengths = torch.normal(mean_tensor, stdev_tensor).to(torch.int)
    lengths = torch.clamp(lengths, min=0, max=max_sequence_length)
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
    lengths[0] = max_sequence_length
    for i in range(batch_size):
        mask[i, : lengths[i]] = 1
    return tokens, lengths, mask


# model_name = "bert-base-uncased"

# hf_model = AutoModel.from_pretrained(model_name).eval()
# bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

BATCH_SIZE = 8
SEQ_LEN = 16
MAX_SEQ_LEN = 256
STD_SEQ_LEN = 10  # let's take a large sequence length
VOCAB_SIZE = 50
N_REPEAT = 10

input_ids, _, attention_mask = get_batch(BATCH_SIZE, SEQ_LEN, MAX_SEQ_LEN, STD_SEQ_LEN, VOCAB_SIZE)

model_name = "facebook/bart-base"

hf_model = AutoModel.from_pretrained(model_name).eval()
bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

_ = bt_model(input_ids, attention_mask)
