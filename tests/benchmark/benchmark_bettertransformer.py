import argparse

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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
    parser.add_argument(
        "--is_decoder",
        action="store_true",
    )
    parser.add_argument(
        "--max_token",
        type=int,
        default=100,
        help="",
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


def timing_cuda(model, num_batches, input_ids, masks, is_decoder, generation_config=None):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_batches):
        if is_decoder:
            _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)
        else:
            _ = model(input_ids, masks)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches


def benchmark(hf_model, bt_model, input_ids, masks, num_batches, is_decoder, max_token, pad_token_id):
    # Warmup
    if is_decoder:
        min_length = max(max_token - 20, 5)

        gen_config = GenerationConfig(
            do_greedy=True,
            max_new_tokens=max_token,
            min_length=min_length,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
        _ = hf_model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
        torch.cuda.synchronize()
        bt_model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
        torch.cuda.synchronize()

    else:
        _ = hf_model(input_ids, masks)
        torch.cuda.synchronize()
        _ = bt_model(input_ids, masks)
        torch.cuda.synchronize()

    # benchmark
    if is_decoder:
        total_hf_time = timing_cuda(hf_model, num_batches, input_ids, masks, is_decoder, gen_config)
        total_bt_time = timing_cuda(bt_model, num_batches, input_ids, masks, is_decoder, gen_config)
    else:
        total_hf_time = timing_cuda(hf_model, num_batches, input_ids, masks, is_decoder)
        total_bt_time = timing_cuda(bt_model, num_batches, input_ids, masks, is_decoder)

    return total_bt_time, total_hf_time


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    BATCH_SIZES = [8, 16, 64]
    SEQ_LEN = [64, 128, 256]
    PAD_PERCENTAGES = [0, 0.1, 0.2, 0.5, 0.75]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.is_decoder:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16 if args.use_half else None, use_cache=True
        ).eval()
    else:
        hf_model = AutoModel.from_pretrained(
            args.model_name, torch_dtype=torch.float16 if args.use_half else None
        ).eval()

    if args.use_cuda:
        hf_model = hf_model.to(0)
    bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

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
                input_ids, _, masks = get_batch(bs, mean_seqlen, max_seqlen, args.seqlen_stdev)

                if args.use_cuda:
                    input_ids = input_ids.to(device)
                    masks = masks.to(device)
                if not args.use_mask:
                    masks = None

                total_bt_time, total_hf_time = benchmark(
                    hf_model,
                    bt_model,
                    input_ids,
                    masks,
                    args.num_batches,
                    args.is_decoder,
                    args.max_token,
                    tokenizer.pad_token_id,
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
