#!/usr/bin/env bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,
# Srikanth Madikeri <srikanth.madikeri@idiap.ch>

echo "$0 $@"

. ./utils/parse_options.sh

if [ $# -ne 2 ]; then
    echo "Initialize a model directory using the same egs of another model. Avoids copying and re-running the same preparation scripts."
    echo "Usage: $0 model_dir_src model_dir_dest"
    exit 1
fi

if [ -f path.sh ]; then
    . path.sh
fi

model_dir_src=$1
model_dir_dest=$2

mkdir -p $model_dir_dest
cp $model_dir_src/{0.trans_mdl,den.fst,feat_dim,normalization.fst,num_pdfs,phones.txt,phone_lm.fst,tree} $model_dir_dest/
model_dir_src_realpath=$(realpath $model_dir_src)
ln -s $model_dir_src_realpath/egs $model_dir_dest/
