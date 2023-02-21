#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"}

# Install test dependencies
pip install pytest pytest-xdist

# Run the test
RUN_SLOW=1 pytest -v -rs tests/onnxruntime/test_trainer.py
