#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"}

# Install docker-compose
# 下载最新版本的 docker-compose 到 /usr/bin 目录下
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Check docker-compose
docker-compose --version

# Build Image
docker build -f Dockerfile-cu10  -t ort/cu10 .
# docker build -f Dockerfile-cu11  -t ort/cu11 .

# Run Image
docker run --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu10:latest $CMD
# docker run -it --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu11:latest $CMD