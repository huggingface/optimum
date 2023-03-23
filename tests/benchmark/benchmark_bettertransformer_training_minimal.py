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
        "--num-epochs",
        type=int,
        default=5,
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


def benchmark_training(model, inputs: Dict, num_epochs: int, batch_size: int):
    num_training_steps = 1024 // batch_size
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
    for _ in range(num_epochs):
        for _ in range(num_training_steps):
            model.zero_grad()
            outputs = model(**inputs)
            loss = outputs.logits.sum()
            loss.backward()

            progress_bar.update(1)
    end_event.record()
    torch.cuda.synchronize()

    max_memory = torch.cuda.max_memory_allocated(device)

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_epochs, max_memory


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32 if args.use_half is False else torch.float16
    hf_model = hf_model.to(device=device, dtype=dtype)

    BATCH_SIZES = [8]
    SEQ_LEN = [1024]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    output_file = open("log_{}_train.csv".format(args.model_name.replace("/", "-")), "w")
    output_file.write(
        "num_epochs, batch_size, seq_len, is cuda, HF time / epoch (s), BT time / epoch (s), Speedup, Eager peak mem (MB), BT peak mem (MB), Mem saving\n"
    )
    num_epochs = args.num_epochs

    for batch_size in BATCH_SIZES:
        for sequence_length in SEQ_LEN:
            print(f"Benchmark on: bs={batch_size}, seq_len={sequence_length}")

            vocab_size = hf_model.config.vocab_size
            inputs = {
                "input_ids": torch.randint(vocab_size - 1, (batch_size, sequence_length), dtype=torch.int64).to(
                    device
                ),
                "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.int64).to(device),
            }

            hf_time_per_epoch, eager_max_mem = benchmark_training(
                hf_model, inputs=inputs, num_epochs=num_epochs, batch_size=batch_size
            )

            bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

            # raise error if no optimized kernel is available
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                bt_time_per_epoch, bt_max_mem = benchmark_training(
                    bt_model, inputs=inputs, num_epochs=num_epochs, batch_size=batch_size
                )

            eager_max_mem = eager_max_mem * 1e-6
            bt_max_mem = bt_max_mem * 1e-6

            print(f"PT eager: {hf_time_per_epoch:.3f} s, peak {eager_max_mem:.2f} MB")
            print(f"PT native: {bt_time_per_epoch:.3f} s, peak {bt_max_mem:.2f} MB")
            speedup = hf_time_per_epoch / bt_time_per_epoch
            mem_saved = eager_max_mem / bt_max_mem

            output_file.write(
                "{},{},{},{},{},{},{}\n".format(
                    num_epochs,
                    batch_size,
                    sequence_length,
                    args.use_cuda,
                    f"{hf_time_per_epoch:.3f}",
                    f"{bt_time_per_epoch:.3f}",
                    f"{speedup:.3f}",
                    f"{eager_max_mem:.3f}",
                    f"{bt_max_mem:.3f}",
                    f"{mem_saved:.3f}",
                )
            )
    output_file.close()
