#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"} 

# Build Image
docker build -f Dockerfile-cu10  -t ort/cu10 .
# docker build -f Dockerfile-cu11  -t ort/cu11 .

# Install nvidia docker toolkits
curl https://nvidia.github.io/nvidia-docker/centos7/nvidia-docker.repo > /etc/yum.repos.d/nvidia-docker.repo
sudo yum update -y && yum install -y nvidia-container-toolkit
sudo systemctl daemon-reload
sudo systemctl restart docker

# Run Image
docker run --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu10:latest $CMD
# docker run -it --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu11:latest $CMD