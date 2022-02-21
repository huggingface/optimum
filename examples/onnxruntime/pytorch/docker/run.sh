#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"} 

# Build Image
docker build -f Dockerfile-cu10  -t ort/cu10 .
# docker build -f Dockerfile-cu11  -t ort/cu11 .

# Install nvidia docker toolkits
# apt install -y nvidia-docker2
# systemctl daemon-reload
# systemctl restart docker
apt-get update && sudo apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Run Image
docker run --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu10:latest $CMD
# docker run -it --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu11:latest $CMD