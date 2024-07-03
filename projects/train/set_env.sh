#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DEEPCLEAN_IFO=H1
export GPU_INDEX=0
export CUDA_LAUNCH_BLOCKING=1
poetry run python -m train --config config-test.yaml
