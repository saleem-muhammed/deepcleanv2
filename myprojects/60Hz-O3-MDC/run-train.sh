#!/bin/bash

export RUN_DIR=$PWD
export DATA_DIR=~/deepclean/data
export RESULTS_DIR=${RUN_DIR}/results
mkdir -p $DATA_DIR $RESULTS_DIR

export CUDA_VISIBLE_DEVICES=0
export DEEPCLEAN_IFO=H1
export GPU_INDEX=0
export CUDA_LAUNCH_BLOCKING=1

cd ../.. 
poetry run law run deepclean.tasks.Train \
    --image train.sif \
    --gpus $GPU_INDEX \
    --data-fname $DATA_DIR/O3_60Hz_train_H1-1250916844-12288.hdf5 \
    --train-config ${RUN_DIR}/config.yaml \
    --output-dir ${RESULTS_DIR} \

cd - 


