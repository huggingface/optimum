import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler
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


def profile_model(model, profile_name, input_ids, masks, num_batches, is_decoder, max_token, pad_token_id):
    # Warmup
    gen_config = None
    if is_decoder:
        gen_config = GenerationConfig(
            max_new_tokens=max_token,
            min_new_tokens=max_token,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
        _ = model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
        torch.cuda.synchronize()

    else:
        _ = model(input_ids, masks)
        torch.cuda.synchronize()

    name = f"{profile_name}_bs={input_ids.shape[0]}_slen={input_ids.shape[1]}_gen={max_token}"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler("./tb_logs/" + name),
    ):
        for _ in tqdm(range(num_batches)):
            if is_decoder:
                with record_function("generate"):
                    _ = model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
            else:
                _ = model(input_ids, masks)

    # prof.export_chrome_trace("./traces/" + name + ".json")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    BATCH_SIZES = [1]
    SEQ_LEN = [64]
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

    bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

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
                    profile_model(
                        hf_model,
                        args.model_name + "_hf_",
                        input_ids,
                        masks,
                        args.num_batches,
                        args.is_decoder,
                        args.max_token,
                        tokenizer.pad_token_id,
                    )

                    # raise error if no optimized kernel is available
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_math=True, enable_mem_efficient=True
                    ):
                        profile_model(
                            bt_model,
                            args.model_name + "_bt_",
                            input_ids,
                            masks,
                            args.num_batches,
                            args.is_decoder,
                            args.max_token,
                            tokenizer.pad_token_id,
                        )
