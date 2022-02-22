#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"} 

# Build Image
# docker build -f Dockerfile-cu10  -t ort/cu10 .
# docker build -f Dockerfile-cu11  -t ort/cu11 .

# Re-install docker
# sudo yum install subscription-manager
# sudo subscription-manager repos --enable rhel-7-server-extras-rpms
# sudo yum install docker -y
# sudo systemctl --now enable docker
# sudo docker -v
# sudo docker run --rm hello-world
# Install nvidia docker toolkits
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
cat /etc/yum.repos.d/nvidia-docker.repo
# sudo yum clean expire-cache
# sudo yum install nvidia-container-toolkit -y
# sudo systemctl daemon-reload
# sudo systemctl restart docker
# sudo docker run --rm -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:11.0-base nvidia-smi
# sudo yum update -y && yum install -y nvidia-container-toolkit
# sudo systemctl daemon-reload
# sudo systemctl restart docker

# Run Image
# docker run --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu10:latest $CMD
# docker run -it --rm -p 80:8888 --gpus $GPU_DEVICES ort/cu11:latest $CMD