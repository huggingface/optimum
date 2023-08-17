import argparse

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from optimum.exporters import TasksManager

from optimum.gptq import load_quantized_model
from accelerate import init_empty_weights

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to benchmark (in the non-quantized case), or reference architecture corresponding to the quantized model (GPTQ case)",
    )
    parser.add_argument(
        "--gptq-model",
        type=str,
        default=None,
        help="Path to a local GPTQ model.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--gptq",
        action='store_true',
        help="Indicate that the model to benchmark is a GPTQ model.",
    )
    parser.add_argument(
        "--sweep",
        action='store_true',
        help="Use the parameter ranges for (batch_size, prompt_length, new_tokens) defined in the .py file instead of the CLI ones.",
    )
    return parser


def timing_cuda(model, num_batches: int, input_ids: torch.Tensor, masks: torch.Tensor, is_decoder: bool, generation_config=None):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    for _ in tqdm(range(num_batches)):
        if is_decoder:
            _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)
        else:
            _ = model(input_ids, masks)
    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)

    return start_event.elapsed_time(end_event) / num_batches, max_memory


def benchmark(model, input_ids: torch.Tensor, masks: torch.Tensor, num_batches: int, is_decoder: bool, new_tokens: int, pad_token_id: int):
    # Warmup
    if is_decoder:
        gen_config = GenerationConfig(
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            use_cache=True,
            pad_token_id=pad_token_id,
            num_beams=1,
            do_sample=False,
        )
        _ = model.generate(input_ids, attention_mask=masks, generation_config=gen_config)
        torch.cuda.synchronize()
    else:
        _ = model(input_ids, masks)
        torch.cuda.synchronize()

    # Benchmark
    if is_decoder:
        total_time, max_mem = timing_cuda(model, num_batches, input_ids, masks, is_decoder, gen_config)
    else:
        total_time, max_mem = timing_cuda(model, num_batches, input_ids, masks, is_decoder)

    return total_time, max_mem


parser = get_parser()
args = parser.parse_args()

if args.sweep:
    batch_sizes = [1, 4, 16, 32]
    prompt_lengths = [512]
    new_tokens = [512]
else:
    batch_sizes = args.batch_size
    prompt_lengths = args.prompt_length
    new_tokens = args.new_tokens


if not torch.cuda.is_available():
    raise ValueError("A cuda device is necessary to benchmark GPTQ.")

device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(args.model)

if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.task:
    task = args.task
else:
    task = TasksManager.infer_task_from_model(args.model)

if task == "text-generation":
    autoclass = AutoModelForCausalLM
elif task == "text2text-generation":
    autoclass = AutoModelForSeq2SeqLM
else:
    autoclass = AutoModel

if task in ["text-generation", "text2text-generation"]:
    is_decoder = True
else:
    is_decoder = False

if args.gptq:
    if not args.gptq_model:
        raise ValueError("The argument --gptq-model needs to be provided when benchmarking GPTQ.")
    
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    empty_model.tie_weights()
    model = load_quantized_model(empty_model, save_folder=args.gptq_model, state_dict_name="model.safetensors", device_map="auto")
else:
    with device:
        model = autoclass.from_pretrained(args.model, torch_dtype=torch.float16)
    
uses_gptq = args.gptq
print(f"Model uses GPTQ: {uses_gptq}")

model = model.eval()

file_name = "log_{}".format(args.model.replace("/", "-"))
if uses_gptq:
    file_name = file_name + "_gptq"
else:
    file_name = file_name + "_nogptq"
file_name = file_name + ".csv"

output_file = open(file_name, "w")
output_file.write(
    "gptq, num_batches, batch_size, prompt_length, new_tokens, Per-token latency (ms), Throughput (tok/s), Max memory (MB)\n"
)

latencies = {}
throughputs = {}
all_max_mem = {}

for batch_size in tqdm(batch_sizes):
    for prompt_length in tqdm(prompt_lengths):
        for new_token in tqdm(new_tokens):
            print(f"---- Running: batch_size={batch_size}, prompt_length={prompt_length}, new_tokens={new_token}")
            
            input_ids = torch.randint(1, model.config.vocab_size - 1, size=(batch_size, prompt_length)).to(device)
            masks = torch.ones(batch_size, prompt_length, dtype=torch.int32).to(device)

            with torch.no_grad():
                mean_latency, max_mem = benchmark(
                    model,
                    input_ids,
                    masks,
                    args.num_batches,
                    is_decoder,
                    new_token,
                    tokenizer.pad_token_id,
                )

            max_mem = max_mem * 1e-6  # in MB
            index = (batch_size, prompt_length, new_token)

            per_token_latency = mean_latency / new_token
            latencies[index] = per_token_latency
            
            throughput = batch_size / (per_token_latency * 1e-3)
            throughputs[index] = throughput
            all_max_mem[index] = max_mem

            # TODO: validate that maxmem is correct
            print(f"Latency per token: {per_token_latency:.3f} ms, throughput: {throughput:.3f} tok/s, peak mem: {max_mem:.2f} MB")

            output_file.write(
                "{},{},{},{},{},{},{},{}\n".format(
                    uses_gptq,
                    args.num_batches,
                    batch_size,
                    prompt_length,
                    new_token,
                    f"{throughput:.4f}",
                    f"{per_token_latency:.4f}",
                    f"{max_mem:.4f}",
                )
            )

output_file.close()
