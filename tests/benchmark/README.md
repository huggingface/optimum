## GPTQ benchmark

Run

```
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model daryl149/llama-2-13b-chat-hf --sweep --num-batches 4 --task text-generation
```

and

```
git clone --branch main https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ
cd Llama-2-13B-chat-GPTQ
mv gptq_model-4bit-128g.safetensors model.safetensors
mv quantize_config.json quantization_config.json

# and then
CUDA_VISIBLE_DEVICES=0 python benchmark_gptq.py --model daryl149/llama-2-13b-chat-hf --gptq-model /path/to/Llama-2-13B-chat-GPTQ/ --sweep --num-batches 4 --gptq --task text-generation
```

