#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"}

# Build Image
# docker build -f Dockerfile-ort1.9.0-cu102  -t ort9/cu10 .
# docker build -f Dockerfile-ort1.9.0-cu111  -t ort9/cu11 .

# Run Image
# docker run -it --rm -p 80:8888 --gpus $GPU_DEVICES ort9/cu10:latest $CMD
# docker run -it --rm -p 80:8888 --gpus $GPU_DEVICES ort9/cu11:latest $CMD

# Install dependencies
pip install transformers datasets
pip install coloredlogs absl-py rouge_score seqeval scipy sacrebleu transformers datasets nltk sklearn
pip install deepspeed mpi4py
python -m unittest tests/onnxruntime/test_onnxruntime_train.py

# Install apex
git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install fairscale
pip install fairscale