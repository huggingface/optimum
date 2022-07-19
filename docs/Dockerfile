FROM nikolaik/python-nodejs:python3.8-nodejs18

RUN apt -y update
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir hf-doc-builder

RUN git clone https://github.com/huggingface/optimum.git
RUN python3 -m pip install --no-cache-dir ./optimum[onnxruntime,benchmark,quality]