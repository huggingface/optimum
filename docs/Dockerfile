FROM nikolaik/python-nodejs:python3.8-nodejs18

ARG commit_sha
ARG clone_url

RUN apt -y update
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir hf-doc-builder

RUN git clone $clone_url && cd optimum && git checkout $commit_sha
RUN python3 -m pip install --no-cache-dir ./optimum[onnxruntime,benchmark,quality]