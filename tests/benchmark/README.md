# BetterTransformer benchmark

Please refer to https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2 & https://pytorch.org/blog/out-of-the-box-acceleration/ for reproduction.

# GPTQ benchmark

Run

```shell
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model daryl149/llama-2-13b-chat-hf --sweep --num-batches 4 --task text-generation
```

and

```shell
git clone --branch main https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ
cd Llama-2-13B-chat-GPTQ
mv gptq_model-4bit-128g.safetensors model.safetensors
mv quantize_config.json quantization_config.json

# and then
# with exllama kernel
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model daryl149/llama-2-13b-chat-hf --gptq-model /path/to/Llama-2-13B-chat-GPTQ/ --sweep --num-batches 4 --gptq --task text-generation

# without exllama kernel
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model daryl149/llama-2-13b-chat-hf --gptq-model /path/to/Llama-2-13B-chat-GPTQ/ --sweep --num-batches 4 --gptq --task text-generation --disable-exllama
```

## Benchmark results

Here are results obtained on a single NVIDIA A100-SXM4-80GB GPU. We use a prompt length of 512, and generate exactly 512 new tokens. Each generation is repeated for 4 batches, and metrics are averaged over the number of batches and generation length.

Additional benchmarks could be done in the act-order case.

From the bencharmk, it appears that Exllama kernel is the best-in-class for GPTQ, although it is rather slow for larger batch sizes. The memory savings are not exactly of x4 although weights are in int4. This can be explained by the possible static buffers used by the kernels, the CUDA context (taken into account in the measurements), and the KV cache that is still in fp16.

### Batch size = 1

|gptq |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|False|None     |None|None      |None  |26.0         |36.958                |27.058            |29152.98        |
|True |False    |4   |128       |exllama|36.2         |33.711                |29.663            |10484.34        |
|True |False    |4   |128       |autogptq-cuda-old|36.2         |46.44                 |21.53             |10344.62        |


### Batch size = 2

|gptq |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|False|None     |None|None      |None  |26.0         |37.35                 |53.53             |30831.09        |
|True |False    |4   |128       |exllama|36.2         |37.25                 |53.68             |12162.43        |
|True |False    |4   |128       |autogptq-cuda-old|36.2         |47.41                 |42.18             |12020.34        |

### Batch size = 4

|gptq |act_order|bits|group_size|kernel           |Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|-----------------|-------------|----------------------|------------------|----------------|
|False|None     |None|None      |None             |26.0         |37.89                 |105.55            |34187.22        |
|True |False    |4   |128       |exllama          |36.2         |54.14                 |73.87             |15518.55        |
|True |False    |4   |128       |autogptq-cuda-old|36.2         |60.98                 |65.59             |15374.67        |


### Batch size = 8

|gptq |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|False|None     |None|None      |None  |26.0         |47.37                 |168.86            |40327.62        |
|True |False    |4   |128       |exllama|36.2         |73.57                 |108.73            |21864.56        |
|True |False    |4   |128       |autogptq-cuda-old|36.2         |104.44                |76.59             |20987.68        |

### Batch size = 16

|gptq |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|False|None     |None|None      |None  |26.0         |69.94                 |228.76            |53986.51        |
|True |False    |4   |128       |exllama|36.2         |95.41                 |167.68            |34777.04        |
|True |False    |4   |128       |autogptq-cuda-old|36.2         |192.48                |83.12             |35497.62        |
