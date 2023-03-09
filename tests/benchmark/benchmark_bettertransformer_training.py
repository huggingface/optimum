import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, get_scheduler

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


WARMUP_STEPS = 2
MAX_STEPS = 20

from datasets import load_dataset
from transformers import AutoTokenizer


def seed_init_fn(x):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def benchmark_training(model, num_epochs: int, train_dataloader, device):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    # warmup
    for _ in range(5):
        batch = next(iter(train_dataloader))
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.logits.sum()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_epochs):
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.logits.sum()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    end_event.record()
    torch.cuda.synchronize()

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_epochs


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_datasets = load_dataset("Abirate/english_quotes", split="train")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32 if args.use_half is False else torch.float16
    hf_model = hf_model.to(device=device, dtype=dtype)

    BATCH_SIZES = [8, 16, 32, 64]
    SEQ_LEN = [32, 64, 128, 256]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    output_file = open("log_{}.csv".format(args.model_name.replace("/", "-")), "w")
    output_file.write("num_epochs, batch_size, seq_len, is cuda, HF time / epoch (s), BT time / epoch (s), Speedup\n")
    num_epochs = args.num_epochs

    for batch_size in BATCH_SIZES:
        for sequence_length in SEQ_LEN:
            print(f"Benchmark on: bs={batch_size}, seq_len={sequence_length}")

            def tokenize_function(example):
                return tokenizer(
                    example["quote"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=sequence_length,
                )

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

            tokenized_datasets = tokenized_datasets.remove_columns(["quote", "author", "tags"])
            tokenized_datasets.set_format("torch")

            train_dataloader = DataLoader(
                tokenized_datasets, shuffle=False, batch_size=batch_size, worker_init_fn=seed_init_fn
            )

            hf_time_per_epoch = benchmark_training(hf_model, num_epochs, train_dataloader, device)

            bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)
            bt_model = bt_model.to(device=device, dtype=dtype)

            bt_time_per_epoch = benchmark_training(
                bt_model,
                num_epochs,
                train_dataloader,
                device,
            )

            speedup = hf_time_per_epoch / bt_time_per_epoch

            output_file.write(
                "{},{},{},{},{},{},{}\n".format(
                    num_epochs,
                    batch_size,
                    sequence_length,
                    args.use_cuda,
                    f"{hf_time_per_epoch:.3f}",
                    f"{bt_time_per_epoch:.3f}",
                    f"{speedup:.3f}",
                )
            )
    output_file.close()
