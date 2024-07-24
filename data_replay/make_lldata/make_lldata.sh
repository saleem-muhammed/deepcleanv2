#!/bin/bash

IFO=H1
# SOURCE=/home/chiajui.chou/dc-demo/data
SOURCE=/home/chiajui.chou/dc-demo/data/O3_AC_clean_H1-1242962000-4096.gwf
START=1242964048
END=1242966096
TAG=llhoft
DESTINATION=/home/chiajui.chou/dc-demo/ll_data/${TAG}_buffer/${IFO}

python make_lldata.py \
    --ifo ${IFO} \
    --source ${SOURCE} \
    --start ${START}\
    --end ${END}\
    --destination ${DESTINATION} \
    --tag ${TAG}