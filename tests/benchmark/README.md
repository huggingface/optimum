# BetterTransformer benchmark

Please refer to https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2 & https://pytorch.org/blog/out-of-the-box-acceleration/ for reproduction.

# GPTQ benchmark

The results below are for AutoGPTQ 0.5.0, PyTorch 2.0.1, bitsandbytes 0.41.1, transformers 4.35.

## Generation benchmark results

Run

```shell
# pytorch fp16
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --sweep --num-batches 4 --task text-generation --generate

# GPTQ with exllamav2 kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 4 --gptq --task text-generation --use-exllama --exllama-version 2 --generate 

# GPTQ with exllama kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 4 --gptq --task text-generation --use-exllama --generate

# GPTQ without exllama kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 4 --gptq --task text-generation --generate

# using bitsandbytes fp4/fp16 scheme
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --sweep --num-batches 4 --task text-generation --bitsandbytes --generate
```

Here are results obtained on a single NVIDIA A100-SXM4-80GB GPU. We use a prompt length of 512, and generate exactly 512 new tokens. Each generation is repeated for 4 batches, and metrics are averaged over the number of batches and generation length.

Additional benchmarks could be done in the act-order case.

From the bencharmk, it appears that Exllama kernel is the best-in-class for GPTQ, although it is rather slow for larger batch sizes. The memory savings are not exactly of x4 although weights are in int4. This can be explained by the possible static buffers used by the kernels, the CUDA context (taken into account in the measurements), and the KV cache that is still in fp16.

Bitsandbytes uses the fp4 scheme, with the compute in fp16.

### Batch size = 1

|quantization |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|None|None     |None|None      |None  |26.0         |36.958                |27.058            |29152.98        |
| gptq | False | 4 | 128 | exllamav2 | 36.07 | 32.25 | 31.01 | 11313.75 |
|gptq |False    |4   |128       |exllama|36.2         |33.711                |29.663            |10484.34        |
|gptq |False    |4   |128       |autogptq-cuda-old|36.2         |46.44                 |21.53             |10344.62        |
|bitsandbytes|None     |None|None      |None  |37.64        |52.00                 |19.23             |11018.36       |

### Batch size = 2

|quantization |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|None|None     |None|None      |None  |26.0         |37.35                 |53.53             |30831.09        |
| gptq | False | 4 | 128 | exllamav2 | 36.07 | 35.81 | 55.85  | 12112.42 |
|gptq |False    |4   |128       |exllama|36.2         |37.25                 |53.68             |12162.43        |
|gptq |False    |4   |128       |autogptq-cuda-old|36.2         |47.41                 |42.18             |12020.34        |
|bitsandbytes|None     |None|None      |None  |37.64        |74.62                 |26.80             |12834.84       |

### Batch size = 4

|quantization |act_order|bits|group_size|kernel           |Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|-----------------|-------------|----------------------|------------------|----------------|
|None|None     |None|None      |None             |26.0         |37.89                 |105.55            |34187.22        |
| gptq | False | 4 | 128 | exllamav2 | 36.07 | 36.04 | 110.98 | 16387.19 |
|gptq |False    |4   |128       |exllama          |36.2         |54.14                 |73.87             |15518.55        |
|gptq |False    |4   |128       |autogptq-cuda-old|36.2         |60.98                 |65.59             |15374.67        |
|bitsandbytes|None     |None|None      |None  |37.64        |80.24                 |49.85             |16187.69       |

### Batch size = 8

|quantization |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|None|None     |None|None      |None  |26.0         |47.37                 |168.86            |40327.62        |
| gptq | False | 4 | 128 | exllamav2 | 36.07 | 47.31 | 169.11 | 22463.02 |
|gptq |False    |4   |128       |exllama|36.2         |73.57                 |108.73            |21864.56        |
|gptq |False    |4   |128       |autogptq-cuda-old|36.2         |104.44                |76.59             |20987.68        |
|bitsandbytes|None     |None|None      |None  |37.64        |91.29                 |87.63             |22894.02       |

### Batch size = 16

|quantization |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|None|None     |None|None      |None  |26.0         |69.94                 |228.76            |53986.51        |
| gptq | False | 4 | 128 | exllamav2 | 36.07 | 83.09 | 192.55 | 35740.95 |
|gptq |False    |4   |128       |exllama|36.2         |95.41                 |167.68            |34777.04        |
|gptq |False    |4   |128       |autogptq-cuda-old|36.2         |192.48                |83.12             |35497.62        |
|bitsandbytes|None     |None|None      |None  |37.64        |113.98                |140.38            |35532.37       |

## Prefill-only benchmark results

Run

```shell
# pytorch fp16
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --sweep --num-batches 10 --task text-generation --prefill --generate

# GPTQ with exllamav2 kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 10 --gptq --task text-generation --prefill --use-exllama --exllama-version 2 --generate 

# GPTQ with exllamav kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 10 --gptq --task text-generation --prefill --use-exllama --generate

# GPTQ without exllama kernel (int4/fp16)
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model TheBloke/Llama-2-13B-chat-GPTQ --sweep --num-batches 10 --gptq --task text-generation --prefill --generate

# using bitsandbytes fp4/fp16 scheme
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model meta-llama/Llama-2-13b-chat-hf --sweep --num-batches 10 --task text-generation --prefill --bitsandbytes --generate
```

The benchmark below is for a prompt length of 512, measuring only the prefill step on a single NVIDIA A100-SXM4-80GB GPU. The forward is repeated 10 times. This benchmark typically corresponds to the forward during training (to the difference that here `generate` is called, which has some overhead).

### Batch size = 1

|quantization |act_order|bits|group_size|kernel           |prompt_length|new_tokens|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Max memory (MB)|
|-----|---------|----|----------|-----------------|-------------|----------|-------------|----------------------|------------------|---------------|
|None|None     |None|None      |None             |512          |1         |27.22        |96.38                 |10.38             |27999.54       |
| gptq | False | 4 | 128 | exllamav2 | 512 | 1 | 6.63 | 116.07  | 8.62  | 10260.35 |
|gptq |False    |4   |128       |exllama          |512          |1         |38.35        |112.54                |8.89              |9330.89        |
|gptq |False    |4   |128       |autogptq-cuda-old|512          |1         |43.94        |368.13                |2.72              |9474.19        |
|bitsandbytes|None|None|None|None|512|1  |37.46|139.17 |7.19 |9952.65 |

### Batch size = 2

|quantization |act_order|bits|group_size|kernel           |prompt_length|new_tokens|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Max memory (MB)|
|-----|---------|----|----------|-----------------|-------------|----------|-------------|----------------------|------------------|---------------|
|None|None     |None|None      |None             |512          |1         |27.22        |169.95                |11.77             |28524.37       |
| gptq | False | 4 | 128 | exllamav2 | 512 | 1 | 6.63 | 212.07  | 9.43  | 10783.60 |
|gptq |False    |4   |128       |exllama          |512          |1         |38.35        |190.44                |10.50             |9855.71        |
|gptq |False    |4   |128       |autogptq-cuda-old|512          |1         |43.94        |443.80                |4.51              |9928.23        |
|bitsandbytes|None|None|None|None|512|1  |37.46|212.76 |9.40 |10421.89|

### Batch size = 4

|quantization |act_order|bits|group_size|kernel           |prompt_length|new_tokens|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Max memory (MB)|
|-----|---------|----|----------|-----------------|-------------|----------|-------------|----------------------|------------------|---------------|
|None|None     |None|None      |None             |512          |1         |27.22        |305.99                |13.07             |29574.01       |
| gptq | False | 4 | 128 | exllamav2 | 512 | 1 | 6.63 | 385.58  | 10.37 | 11829.59 |
|gptq |False    |4   |128       |exllama          |512          |1         |38.35        |345.54                |11.58             |10905.35       |
|gptq |False    |4   |128       |autogptq-cuda-old|512          |1         |43.94        |597.24                |6.70              |10838.42       |
|bitsandbytes|None|None|None|None|512|1  |37.46|349.18 |11.46|11440.08|

### Batch size = 8

|quantization |act_order|bits|group_size|kernel           |prompt_length|new_tokens|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Max memory (MB)|
|-----|---------|----|----------|-----------------|-------------|----------|-------------|----------------------|------------------|---------------|
|None|None     |None|None      |None             |512          |1         |27.22        |600.47                |13.32             |31673.30       |
| gptq | False | 4 | 128 | exllamav2 | 512 | 1 | 6.63 | 753.06  | 10.62 | 13920.50 |
|gptq |False    |4   |128       |exllama          |512          |1         |38.35        |659.61                |12.13             |13004.64       |
|gptq |False    |4   |128       |autogptq-cuda-old|512          |1         |43.94        |909.09                |8.80              |12862.18       |
|bitsandbytes|None|None|None|None|512|1  |37.46|643.42 |12.43|13539.37|

### Batch size = 16

|quantization |act_order|bits|group_size|kernel    |prompt_length|new_tokens|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Max memory (MB)|
|-----|---------|----|-----------|----------|-------------|----------|-------------|----------------------|------------------|---------------|
|None|None     |None|None      |None        |512          |1         |27.22        |1209.07               |13.23             |35871.88       |
| gptq | False | 4 | 128 | exllamav2 | 512 | 1 | 6.63 | 1467.36 | 10.90 | 18104.44 |
|gptq |False    |4   |128       |exllama     |512          |1         |38.35        |1280.25               |12.50             |17203.22       |
|gptq |False    |4   |128       |autogptq-cuda-old |512          |1         |43.94        |1533.54               |10.43             |17060.76       |
|bitsandbytes|None|None|None|None|512|1  |37.46|1256.88|12.73|17737.95|

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