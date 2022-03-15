#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"}

# Build Image
# docker build -f Dockerfile-cu10  -t ort/cu10 .
# docker build -f Dockerfile-cu11  -t ort/cu11 .

# Run Image
# docker run -it --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu10:latest $CMD
# docker run -it --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu11:latest $CMD

# Install dependencies
pip install coloredlogs transformers>=4.15.0 datasets>=1.8.0
pip install scipy sklearn
python -m unittest tests/onnxruntime/test_onnxruntime_train.py
