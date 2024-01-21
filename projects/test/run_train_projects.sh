#!/bin/bash

export DEEPCLEAN_IFO=K1
export GPU_INDEX=0
poetry run python -m train --config config_projects.yaml
