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

### Batch size = 2

### Batch size = 4

### Batch size = 8

### Batch size = 16


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

### Batch size = 2

### Batch size = 4

### Batch size = 8

### Batch size = 16


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