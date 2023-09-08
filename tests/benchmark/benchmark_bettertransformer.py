import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from optimum.bettertransformer import BetterTransformer
from optimum.exporters import TasksManager


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
        help="True average sequence length (the rest will be padding).",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=256,
        help="Input padded sequence length.",
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
    parser.add_argument("--is_decoder", action="store_true", help="Benchmark the generate method.")
    parser.add_argument(
        "--sweep",
        action="store_true",
    )
    parser.add_argument(
        "--max_token",
        type=int,
        default=100,
        help="Number of new tokens, for autoregressive models using generate.",
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
    if is_decoder:
        model.generation_config.eos_token_id = None

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # We need NOT call torch.cuda.empty_cache() here as it appears to negate the warmup.

    latencies = []
    for _ in tqdm(range(num_batches)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        if is_decoder:
            _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)
        else:
            _ = model(input_ids, masks)
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        print(f"\nLatency per token: {latency_ms / generation_config.min_new_tokens:.3f} ms")
        latencies.append(latency_ms)

    max_memory = torch.cuda.max_memory_allocated(device)

    return np.mean(latencies), max_memory


def benchmark(model, input_ids, masks, num_batches, is_decoder, max_token, pad_token_id):
    if is_decoder:
        gen_config = GenerationConfig(
            max_new_tokens=max_token,
            min_new_tokens=max_token,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=None,
        )

    # warmup
    if is_decoder:
        _ = model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
        torch.cuda.synchronize()
    else:
        _ = model(input_ids, masks)
        torch.cuda.synchronize()

    # benchmark
    if is_decoder:
        total_time, max_mem = timing_cuda(model, num_batches, input_ids, masks, is_decoder, gen_config)
    else:
        total_time, max_mem = timing_cuda(model, num_batches, input_ids, masks, is_decoder)

    return total_time, max_mem


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.sweep:
        BATCH_SIZES = [1, 2, 4]
        SEQ_LEN = [64, 128]
    else:
        BATCH_SIZES = [args.batch_size]
        SEQ_LEN = [args.max_seqlen]

    if args.is_decoder:
        PAD_PERCENTAGES = [0]
    else:
        PAD_PERCENTAGES = [0, 0.1, 0.2, 0.5, 0.75]

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task = TasksManager.infer_task_from_model(args.model_name)

    if task == "text-generation":
        autoclass = AutoModelForCausalLM
    elif task == "text2text-generation":
        autoclass = AutoModelForSeq2SeqLM
    else:
        autoclass = AutoModel

    if args.use_cuda:
        with torch.device("cuda:0"):
            hf_model = autoclass.from_pretrained(args.model_name, torch_dtype=torch.float16 if args.use_half else None)
        # in PyTorch we trust :)
        hf_model = hf_model.to("cuda:0")
        hf_model = hf_model.to(torch.float16)
    else:
        hf_model = autoclass.from_pretrained(args.model_name, torch_dtype=torch.float16 if args.use_half else None)

    output_name = "log_{}.csv".format(args.model_name.replace("/", "-"))
    output_file = open(output_name, "w")
    output_file.write(
        "num_batches, batch_size, seq_len, is cuda, is half, use mask, pad percentage, Latency eager (ms), Latency BT (ms), Speedup (%), Mem eager (MB), Mem BT (MB), Mem saved (%)\n"
    )

    all_total_hf_time = {}
    all_max_mem_eager = {}

    for bs in tqdm(BATCH_SIZES):
        for seq_len in tqdm(SEQ_LEN):
            for pad_perc in tqdm(PAD_PERCENTAGES):
                print(f"-- Running: bs={bs}, seq_len={seq_len}")
                # current_std = int(seq_len*pad_perc)
                # max_seqlen = seq_len + current_std
                max_seqlen = seq_len
                mean_seqlen = int((1 - pad_perc) * max_seqlen)
                input_ids, _, masks = get_batch(
                    bs, mean_seqlen, max_seqlen, args.seqlen_stdev, vocab_size=hf_model.config.vocab_size
                )

                if args.use_cuda:
                    input_ids = input_ids.to(device)
                    masks = masks.to(device)

                if args.use_mask is False and bs == 1:
                    masks = None

                with torch.inference_mode():
                    total_hf_time, max_mem_eager = benchmark(
                        hf_model,
                        input_ids,
                        masks,
                        args.num_batches,
                        args.is_decoder,
                        args.max_token,
                        tokenizer.pad_token_id,
                    )

            all_total_hf_time[(bs, seq_len)] = total_hf_time
            all_max_mem_eager[(bs, seq_len)] = max_mem_eager

    bt_model = BetterTransformer.transform(hf_model)
    for bs in tqdm(BATCH_SIZES):
        for seq_len in tqdm(SEQ_LEN):
            for pad_perc in tqdm(PAD_PERCENTAGES):
                print(f"-- Running: bs={bs}, seq_len={seq_len}")
                max_seqlen = seq_len
                mean_seqlen = int((1 - pad_perc) * max_seqlen)
                input_ids, _, masks = get_batch(
                    bs, mean_seqlen, max_seqlen, args.seqlen_stdev, vocab_size=hf_model.config.vocab_size
                )

                if args.use_cuda:
                    input_ids = input_ids.to(device)
                    masks = masks.to(device)

                if args.use_mask is False and bs == 1:
                    masks = None

                with torch.inference_mode():
                    # raise error if no optimized kernel is available
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_math=True, enable_mem_efficient=True
                    ):
                        total_bt_time, max_mem_bt = benchmark(
                            bt_model,
                            input_ids,
                            masks,
                            args.num_batches,
                            args.is_decoder,
                            args.max_token,
                            tokenizer.pad_token_id,
                        )

                total_hf_time = all_total_hf_time[(bs, seq_len)]
                max_mem_eager = all_max_mem_eager[(bs, seq_len)]

                speedup = (total_hf_time / total_bt_time - 1) * 100
                mem_saved = (max_mem_eager / max_mem_bt - 1) * 100

                max_mem_eager = max_mem_eager * 1e-6
                max_mem_bt = max_mem_bt * 1e-6

                print(f"PT eager: {total_hf_time:.3f} ms, peak {max_mem_eager:.2f} MB")
                print(f"PT native: {total_bt_time:.3f} ms, peak {max_mem_bt:.2f} MB")

                output_file.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        args.num_batches,
                        bs,
                        seq_len,
                        args.use_cuda,
                        args.use_half,
                        args.use_mask,
                        pad_perc,
                        f"{total_hf_time:.3f}",
                        f"{total_bt_time:.3f}",
                        f"{speedup:.3f}",
                        f"{max_mem_eager:.3f}",
                        f"{max_mem_bt:.3f}",
                        f"{mem_saved:.3f}",
                    )
                )

    output_file.close()
    print("RESULTS:")
    df = pd.read_csv(output_name)
    print(df.to_markdown(index=False))
