#!/bin/bash

CMD=${1:-/bin/bash}
GPU_DEVICES=${2:-"all"}

# Run the test
RUN_SLOW=1 pytest -v -rs test_trainer.py
