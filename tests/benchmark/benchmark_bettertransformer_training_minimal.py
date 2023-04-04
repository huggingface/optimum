import argparse
import random
from typing import Dict

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from optimum.bettertransformer import BetterTransformer


torch.backends.cuda.matmul.allow_tf32 = True


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=100,
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="hf-internal-testing/tiny-random-gpt2",
        help="",
    )

    parser.add_argument(
        "--use-half",
        action="store_true",
    )

    parser.add_argument(
        "--use-cuda",
        action="store_true",
    )

    return parser


def seed_init_fn(x):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def benchmark_training(model, inputs: Dict, num_training_steps: int):
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    # warmup
    for _ in range(5):
        outputs = model(**inputs)
        loss = outputs.logits.sum()
        loss.backward()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_training_steps):
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.logits.sum()
        loss.backward()

        progress_bar.update(1)
    end_event.record()
    torch.cuda.synchronize()

    max_memory = torch.cuda.max_memory_allocated(device)

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_training_steps, max_memory


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with torch.device(device):
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16 if args.use_half else None
        )
    hf_model = hf_model.to(device)

    BATCH_SIZES = [8]
    SEQ_LEN = [1024]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    output_file = open("log_{}_train.csv".format(args.model_name.replace("/", "-")), "w")
    output_file.write(
        "num_training_steps, batch_size, seq_len, is cuda, Time per batch (eager - s), Time per batch (BT - s), Speedup (%), Eager peak mem (MB), BT peak mem (MB), Mem saving (%)\n"
    )
    all_hf_time_per_batch = {}
    all_eager_max_mem = {}

    for batch_size in BATCH_SIZES:
        for sequence_length in SEQ_LEN:
            print(f"Benchmark PT on: bs={batch_size}, seq_len={sequence_length}")

            vocab_size = hf_model.config.vocab_size
            inputs = {
                "input_ids": torch.randint(vocab_size - 1, (batch_size, sequence_length), dtype=torch.int64).to(
                    device
                ),
                "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.int64).to(device),
            }

            hf_time_per_batch, eager_max_mem = benchmark_training(
                hf_model, inputs=inputs, num_training_steps=args.num_training_steps
            )

            all_hf_time_per_batch[(batch_size, sequence_length)] = hf_time_per_batch
            all_eager_max_mem[(batch_size, sequence_length)] = eager_max_mem

    bt_model = BetterTransformer.transform(hf_model)
    for batch_size in BATCH_SIZES:
        for sequence_length in SEQ_LEN:
            print(f"Benchmark BT on: bs={batch_size}, seq_len={sequence_length}")

            vocab_size = hf_model.config.vocab_size
            inputs = {
                "input_ids": torch.randint(vocab_size - 1, (batch_size, sequence_length), dtype=torch.int64).to(
                    device
                ),
                "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.int64).to(device),
            }

            # raise error if no optimized kernel is available
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                bt_time_per_batch, bt_max_mem = benchmark_training(
                    bt_model, inputs=inputs, num_training_steps=args.num_training_steps
                )

            eager_max_mem = all_eager_max_mem[(batch_size, sequence_length)] * 1e-6
            bt_max_mem = bt_max_mem * 1e-6

            hf_time_per_batch = all_hf_time_per_batch[(batch_size, sequence_length)]

            print(f"PT eager: {hf_time_per_batch:.3f} s, peak {eager_max_mem:.2f} MB")
            print(f"PT native: {bt_time_per_batch:.3f} s, peak {bt_max_mem:.2f} MB")
            speedup = (hf_time_per_batch / bt_time_per_batch - 1) * 100
            mem_saved = (eager_max_mem / bt_max_mem - 1) * 100

            output_file.write(
                "{},{},{},{},{},{},{},{},{},{}\n".format(
                    args.num_training_steps,
                    batch_size,
                    sequence_length,
                    args.use_cuda,
                    f"{hf_time_per_batch:.3f}",
                    f"{bt_time_per_batch:.3f}",
                    f"{speedup:.3f}",
                    f"{eager_max_mem:.3f}",
                    f"{bt_max_mem:.3f}",
                    f"{mem_saved:.3f}",
                )
            )

    output_file.close()
