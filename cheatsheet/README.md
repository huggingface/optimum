# Optimization tools for production use 🤗

We present here a set of general optimisation tools that can be used for faster, lighter training and inference and how to use it in 🤗 ecosystem if applicable!

## Table of contents

**Sorted in alphabetical order**

## Model shrinking

Here we present a set of tools to reduce the size of your models, termed as model shrinking.

### DeepSpeed inference

<img src="https://github.com/microsoft/DeepSpeed/blob/master/docs/assets/images/deepspeed-logo-uppercase-white.svg" width="512"/>


DeepSpeed provides a set of optimization tools for training and inference, and proposes ligther inference tools using int8 and int4 quantization. 

- [:octocat: GitHub repo](https://github.com/microsoft/DeepSpeed)
- [🤗 integration](https://huggingface.co/docs/transformers/main_classes/deepspeed#deepspeed-integration)
- **Targetted hardwares**: GPU
- **NEW:** *int4 quantization*: https://github.com/microsoft/DeepSpeed/pull/2526 

### int8 `bitsandbytes` quantization

<img src="https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Thumbnail_blue.png" width="512"/>

`bitandbytes` integrates the 2-stages quantization method proposed in [LLM.int8(): matrix multiplication at scale](https://arxiv.org/abs/2208.07339). A 🤗 integration is also available for most of the models (text, audio, vision).

- [:octocat: GitHub repo](https://github.com/TimDettmers/bitsandbytes)
- Google colab demo: [![Colab demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4#scrollTo=W8tQtyjp75O_)
- [![Youtube demo]("../assets/logos/youtube-music.png")](https://www.youtube.com/watch?v=lI3bZzsQcjs)
- [:closed_book: Blogpost](https://huggingface.co/blog/hf-bitsandbytes-integration)
- **Targetted hardwares**: GPU

### Intel Neural Compressor

### ONNX Runtime Quantization

![image](https://github.com/microsoft/onnxruntime/blob/main/docs/images/ONNX_Runtime_logo_dark.png)

- [:octocat: GitHub repo](https://github.com/microsoft/onnxruntime)
- [🤗 integration](https://huggingface.co/docs/optimum/onnxruntime/overview)

## Faster Inference 

### AI Template

## Faster Training

### ONNX Runtime
