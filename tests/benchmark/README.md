# BetterTransformer benchmark

Please refer to https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2 & https://pytorch.org/blog/out-of-the-box-acceleration/ for reproduction.

# GPTQ benchmark

The results below are for AutoGPTQ 0.7.0, PyTorch 2.2.0, bitsandbytes 0.42.0, transformers 4.37.2.

Here are results obtained on a single NVIDIA A100-SXM4-80GB GPU **without act-order**. Additional benchmarks could be done in the act-order case.

From the benchmark, it appears that Exllama kernel is the best-in-class for GPTQ, although it is rather slow for larger batch sizes. The memory savings are not exactly of x4 although weights are in int4. This can be explained by the possible static buffers used by the kernels, the CUDA context (taken into account in the measurements), and the KV cache that is still in fp16.

Bitsandbytes uses the fp4 scheme, with the compute in fp16.

**Beware that exllama uses [fp16 accumulation](https://github.com/turboderp/exllamav2/blob/75f969a6d3efd28fcb521100669ba2594f3ba14c/exllamav2/exllamav2_ext/cuda/q_gemm.cu#L132-L138) for its fp16 x fp16 GEMM, while PyTorch and other kernels accumulate on fp32 for numerical accuracy purposees. This has latency implications and the comparison is therefore not apple-to-apple.**

## Prefill benchmark

The benchmark below is for a prompt length of 512, measuring only the prefill step on a single NVIDIA A100-SXM4-80GB GPU. This benchmark typically corresponds to the forward during training (to the difference that here `generate` is called, which has some overhead).

Run

```shell
# pytorch fp16
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --sweep --task text-generation --generate --prefill

# GPTQ with exllamav2 kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --gptq --task text-generation --use-exllama --exllama-version 2 --generate --prefill

# GPTQ with exllama kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep -gptq --task text-generation --use-exllama --generate --prefill

# GPTQ without exllama kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --gptq --task text-generation --generate --prefill

# GPTQ with marlin kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --gptq --task text-generation --use-marlin --generate --prefill

# using bitsandbytes fp4/fp16 scheme
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --sweep --bitsandbytes --task text-generation --generate --prefill
```

### Batch size = 1

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 112.08        | 98.89                  | 10.11              |
| gptq         | False     | 4    | 128        | cuda-old   | 6.09          | 374.60                 | 2.67               |
| gptq         | False     | 4    | 128        | exllama    | 5.99          | 116.11                 | 8.61               |
| gptq         | False     | 4    | 128        | exllama_v2 | 7.28          | 115.05                 | 8.69               |
| gptq         | False     | 4    | 128        | marlin     | 32.26         | 95.15                  | 10.51              |
| bitsandbytes | None      | None | None       | None       | 10.18         | 140.90                 | 7.10               |
### Batch size = 2

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 112.08        | 183.41                 | 10.90              |
| gptq         | False     | 4    | 128        | cuda-old   | 6.09          | 458.15                 | 4.37               |
| gptq         | False     | 4    | 128        | exllama    | 5.99          | 196.50                 | 10.18              |
| gptq         | False     | 4    | 128        | exllama_v2 | 7.28          | 195.30                 | 10.24              |
| gptq         | False     | 4    | 128        | marlin     | 32.26         | 192.18                 | 10.41              |
| bitsandbytes | None      | None | None       | None       | 10.18         | 223.30                 | 8.96               |

### Batch size = 4

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 112.08        | 332.39                 | 12.03              |
| gptq         | False     | 4    | 128        | cuda-old   | 6.09          | 618.96                 | 6.46               |
| gptq         | False     | 4    | 128        | exllama    | 5.99          | 353.67                 | 11.31              |
| gptq         | False     | 4    | 128        | exllama_v2 | 7.28          | 353.47                 | 11.32              |
| gptq         | False     | 4    | 128        | marlin     | 32.26         | 384.47                 | 10.40              |
| bitsandbytes | None      | None | None       | None       | 10.18         | 369.76                 | 10.82              |

### Batch size = 8

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 112.08        | 655.58                 | 12.20              |
| gptq         | False     | 4    | 128        | cuda-old   | 6.09          | 962.64                 | 8.31               |
| gptq         | False     | 4    | 128        | exllama    | 5.99          | 687.99                 | 11.63              |
| gptq         | False     | 4    | 128        | exllama_v2 | 7.28          | 684.68                 | 11.68              |
| gptq         | False     | 4    | 128        | marlin     | 32.26         | 760.58                 | 10.52              |
| bitsandbytes | None      | None | None       | None       | 10.18         | 689.23                 | 11.61              |

### Batch size = 16

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 112.08        | 1368.83                | 11.69              |
| gptq         | False     | 4    | 128        | cuda-old   | 6.09          | 1679.88                | 9.52               |
| gptq         | False     | 4    | 128        | exllama    | 5.99          | 1337.64                | 11.96              |
| gptq         | False     | 4    | 128        | exllama_v2 | 7.28          | 1336.79                | 11.97              |
| gptq         | False     | 4    | 128        | marlin     | 32.26         | 1515.79                | 10.56              |
| bitsandbytes | None      | None | None       | None       | 10.18         | 1427.68                | 11.21              |

## Decode benchmark

The benchmark below is for a prefill length of 1, essentially measuring the decode step in text generation (512 tokens generated).

Run

```shell
# pytorch fp16
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --sweep --num-batches 5 --task text-generation --generate --decode

# GPTQ with exllamav2 kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 5 --gptq --task text-generation --use-exllama --exllama-version 2 --generate --decode

# GPTQ with exllama kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 5 --gptq --task text-generation --use-exllama --exllama-version 1 --generate --decode

# GPTQ with cuda-old kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 5 --gptq --task text-generation --generate --decode

# GPTQ with marlin kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 5 --gptq --task text-generation --use-marlin --generate --decode

# using bitsandbytes fp4/fp16 scheme
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --sweep --num-batches 5 --bitsandbytes --task text-generation --generate --decode
```

### Batch size = 1

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 6.64          | 30.43                  | 32.86              |
| gptq         | False     | 4    | 128        | cuda-old   | 6.03          | 42.91                  | 23.30              |
| gptq         | False     | 4    | 128        | exllama    | 6.65          | 31.68                  | 31.57              |
| gptq         | False     | 4    | 128        | exllama_v2 | 5.86          | 31.60                  | 31.64              |
| gptq         | False     | 4    | 128        | marlin     | 31.75         | 28.96                  | 34.53              |
| bitsandbytes | None      | None | None       | None       | 9.80          | 45.06                  | 22.19              |

### Batch size = 2

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 6.64          | 30.11                  | 66.42              |
| gptq         | False     | 4    | 128        | cuda-old   | 6.03          | 42.68                  | 46.86              |
| gptq         | False     | 4    | 128        | exllama    | 6.65          | 37.00                  | 54.05              |
| gptq         | False     | 4    | 128        | exllama_v2 | 5.86          | 31.74                  | 63.02              |
| gptq         | False     | 4    | 128        | marlin     | 31.75         | 29.19                  | 68.53              |
| bitsandbytes | None      | None | None       | None       | 9.80          | 68.00                  | 29.41              |

### Batch size = 4

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 6.64          | 29.76                  | 134.41             |
| gptq         | False     | 4    | 128        | cuda-old   | 6.03          | 51.43                  | 77.78              |
| gptq         | False     | 4    | 128        | exllama    | 6.65          | 55.15                  | 72.53              |
| gptq         | False     | 4    | 128        | exllama_v2 | 5.86          | 31.58                  | 126.68             |
| gptq         | False     | 4    | 128        | marlin     | 31.75         | 29.08                  | 137.56             |
| bitsandbytes | None      | None | None       | None       | 9.80          | 70.25                  | 56.94              |

### Batch size = 8

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 6.64          | 32.98                  | 242.60             |
| gptq         | False     | 4    | 128        | cuda-old   | 6.03          | 91.74                  | 87.20              |
| gptq         | False     | 4    | 128        | exllama    | 6.86          | 58.61                  | 136.49             |
| gptq         | False     | 4    | 128        | exllama_v2 | 5.86          | 32.59                  | 245.48             |
| gptq         | False     | 4    | 128        | marlin     | 31.75         | 29.02                  | 275.70             |
| bitsandbytes | None      | None | None       | None       | 9.80          | 74.20                  | 107.81             |

### Batch size = 16

| quantization | act_order | bits | group_size | kernel     | Load time (s) | Per-token latency (ms) | Throughput (tok/s) |
|--------------|-----------|------|------------|------------|---------------|------------------------|--------------------|
| None         | None      | None | None       | None       | 6.64          | 40.24                  | 397.61             |
| gptq         | False     | 4    | 128        | cuda-old   | 6.03          | 171.90                 | 93.08              |
| gptq         | False     | 4    | 128        | exllama    | 6.86          | 66.37                  | 241.07             |
| gptq         | False     | 4    | 128        | exllama_v2 | 5.86          | 48.10                  | 332.61             |
| gptq         | False     | 4    | 128        | marlin     | 31.75         | 31.71                  | 504.63             |
| bitsandbytes | None      | None | None       | None       | 9.80          | 82.29                  | 194.44             |

## Perplexity benchmark results

Run

```shell
# pytorch fp16
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --task text-generation --ppl

# GPTQ with exllamav2 kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --revision gptq-4bit-128g-actorder_True --gptq --task text-generation --use-exllama --exllama-version 2 --ppl

# GPTQ with exllama kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --revision gptq-4bit-128g-actorder_True --gptq  --task text-generation --use-exllama --ppl

# GPTQ without exllama kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --revision gptq-4bit-128g-actorder_True --gptq --task text-generation --ppl

# using bitsandbytes fp4/fp16 scheme
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf ---task text-generation --bitsandbytes --ppl
```

| quantization | act_order | bits | group_size | kernel           | perplexity |
|--------------|-----------|------|------------|------------------|------------|
| None         | None      | None | None       | None             | 6.61       |
| gptq         | True      | 4    | 128        | exllamav2        | 6.77       |
| gptq         | True      | 4    | 128        | exllama          | 6.77       |
| gptq         | True      | 4    | 128        | autogptq-cuda-old| 6.77       |
| bitsandbytes | None      | 4    | None       | None             | 6.78       |