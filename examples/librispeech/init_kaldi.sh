#!/bin/bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,


ln -s ../../path.sh ./path.sh
. ./path.sh

if [ $# -lt 1 ]; then
    if [ -z $KALDI_ROOT ]; then
        echo "$0: KALDI_ROOT is not defined and no argument found to the script".
        exit 1
    fi
else
    KALDI_ROOT=$1
fi

ln -s $KALDI_ROOT/egs/librispeech/s5/{utils,steps,local} .

for dir in misc src shutil cmd.sh config; do 
    ln -s ../../$dir ./
done
