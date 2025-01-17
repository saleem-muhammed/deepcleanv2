#!/bin/bash

IFO=L1
CHANNELS=/home/chiajui.chou/ll_data_O4_review/get_lldata/chanslist_O3.txt
START=1250916945
DURATION=12
LENGTH=12288
DESTINATION=/home/chiajui.chou/ll_data_O4_review/unresampled_data
HOFT_TAG=HOFT
DETCHAR_TAG=INMON
KIND=lldetchar

python get_data.py\
    --ifo ${IFO}\
    --channels ${CHANNELS}\
    --start ${START}\
    --duration ${DURATION}\
    --length ${LENGTH}\
    --destination ${DESTINATION}\
    --hoft_tag ${HOFT_TAG}\
    --detchar_tag ${DETCHAR_TAG}\
    --kind ${KIND}