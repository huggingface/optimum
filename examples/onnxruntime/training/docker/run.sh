#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"}

# Run the test
python -m unittest tests/onnxruntime/test_trainer.py
