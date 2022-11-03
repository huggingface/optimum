import argparse

import torch
from transformers import AutoModel

from optimum.bettertransformer import BetterTransformer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="",
    )
    parser.add_argument(
        "--avg-seqlen",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="",
    )
    parser.add_argument(
        "--seqlen-stdev",
        type=int,
        default=10,
        help="",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
    )
    parser.add_argument(
        "--use-half",
        action="store_true",
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
    )
    return parser


def get_batch(batch_size, avg_seqlen, max_sequence_length, seqlen_stdev, vocab_size=30522, pad_idx=0):
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
    # lengths[0:2] = max_sequence_length-1
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


def timing_cuda(model, num_batches, input_ids, masks):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_batches):
        _ = model(input_ids, masks)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches


def benchmark(model_name, num_batches, batch_size, avg_seqlen, max_seqlen, seqlen_stdev, is_cuda, is_half, use_mask):
    print("Loading model {}".format(model_name))
    hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if is_half else None).eval()
    if is_cuda:
        hf_model = hf_model.to(0)
    bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

    input_ids, _, masks = get_batch(batch_size, avg_seqlen, max_seqlen, seqlen_stdev)

    if is_cuda:
        input_ids = input_ids.to(0)
        masks = masks.to(0)

    if not use_mask:
        masks = None

    # Warmup
    _ = hf_model(input_ids[0].unsqueeze(0), masks[0].unsqueeze(0))
    torch.cuda.synchronize()
    _ = bt_model(input_ids[0].unsqueeze(0), masks[0].unsqueeze(0))
    torch.cuda.synchronize()

    total_hf_time = timing_cuda(hf_model, num_batches, input_ids, masks)
    total_bt_time = timing_cuda(bt_model, num_batches, input_ids, masks)

    return total_bt_time, total_hf_time


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    BATCH_SIZES = [8, 16, 64]
    SEQ_LEN = [64, 128, 256]
    PAD_PERCENTAGES = [0, 0.1, 0.2, 0.5, 0.75]

    output_file = open("log_{}.csv".format(args.model_name.replace("/", "-")), "w")
    output_file.write(
        "num_batches, batch_size, seq_len, is cuda, is half, use mask, pad percentage, HF time, BT time, Speedup\n"
    )
    for bs in BATCH_SIZES:
        for seq_len in SEQ_LEN:
            for pad_perc in PAD_PERCENTAGES:
                # current_std = int(seq_len*pad_perc)
                # max_seqlen = seq_len + current_std
                max_seqlen = seq_len
                mean_seqlen = int((1 - pad_perc) * max_seqlen)

                total_bt_time, total_hf_time = benchmark(
                    args.model_name,
                    args.num_batches,
                    bs,
                    mean_seqlen,
                    max_seqlen,
                    args.seqlen_stdev,
                    args.use_cuda,
                    args.use_half,
                    args.use_mask,
                )

                speedup = total_hf_time / total_bt_time

                output_file.write(
                    "{},{},{},{},{},{},{},{},{},{}\n".format(
                        args.num_batches,
                        args.use_cuda,
                        bs,
                        seq_len,
                        args.use_half,
                        args.use_mask,
                        pad_perc,
                        total_hf_time,
                        total_bt_time,
                        speedup,
                    )
                )
    output_file.close()
    # print(total_bt_time, total_hf_time)
    # print("BT w.r.t HF: {}".format(total_hf_time/total_bt_time))
