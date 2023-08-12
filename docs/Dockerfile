FROM nikolaik/python-nodejs:python3.8-nodejs18

ARG commit_sha
ARG clone_url

RUN apt -y update
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN echo "aaa"
RUN python3 -m pip install --no-cache-dir git+https://github.com/regisss/doc-builder.git@test_gptq_doc

RUN git clone https://github.com/huggingface/optimum.git && cd optimum && git checkout fix_doc_gptq
RUN python3 -m pip install --no-cache-dir ./optimum[onnxruntime,benchmark,quality,exporters-tf,doc-build]
