#!/bin/bash

export DEEPCLEAN_IFO=H1
export GPU_INDEX=0
poetry run law run deepclean.tasks.Train \
	--image train.sif \
	--gpus $GPU_INDEX \
	--data-fname $DATA_DIR/H-H1_lldata-1369291863-16384.hdf5 \
	--output-dir $RESULTS_DIR/my-first-run_H1
