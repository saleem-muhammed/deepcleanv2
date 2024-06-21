#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DEEPCLEAN_IFO=H1
export GPU_INDEX=0
export CUDA_LAUNCH_BLOCKING=1
poetry run law run deepclean.tasks.Train \
    --image train.sif \
    --gpus $GPU_INDEX \
    --data-fname $DATA_DIR/O3_AC_clean_H1-1242962000-4096.hdf5 \
    --train-config ${HOME}/deepcleanv2/projects/train/config-test.yaml \
    --output-dir ${RESULTS_DIR}/test-O3-H1 \
