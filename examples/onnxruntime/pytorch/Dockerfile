# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
CMD nvidia-smi
WORKDIR /home/ubuntu/workdir

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# Bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake
RUN unattended-upgrade
RUN apt-get autoremove -y

# Install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

ENV PATH /opt/conda/bin:$PATH

# Create the environment:
# Python and pip
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y  python-mpi4py
COPY ort_train_env.yml .
RUN conda update conda \
	&& conda env create --name venv_ort -f ort_train_env.yml

RUN echo "source activate venv_ort" > ~/.bashrc
ENV PATH /opt/conda/envs/venv_ort/bin:$PATH
ENV CONDA_DEFAULT_ENV $venv_ort

# PyTorch
RUN pip install onnx==1.9.0 ninja
RUN pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# ORT Module
RUN pip install onnxruntime-training==1.9.0

RUN pip install torch-ort
RUN python -m torch_ort.configure

# Clone the optimum repo on branch `ort-train`
RUN git clone https://github.com/huggingface/optimum.git && cd optimum && git checkout ort-trainer && cp tests/onnxruntime/test_onnxruntime_trainer.py ./

CMD ["/bin/bash"]
