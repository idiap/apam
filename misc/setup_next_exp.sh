#!/bin/bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,
# Srikanth Madikeri <srikanth.madikeri@idiap.ch>


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 old_exp_dir new_exp_dir"
    echo ""
    exit 1
fi
src=$1     # path to old experiment dir to copy from
dest=$2    # path to new experiment dir to copy to

echo "$0 $@"

mkdir -p $dest
fullpath=`realpath $src`

ln -s $fullpath/egs $dest/egs

for dir in configs tree init; do
    cp -r $src/$dir $dest/$dir
done

for file in 0.trans_mdl den.fst normalization.fst phone_lm.fst feat_dim num_pdfs phones.txt; do
    cp $src/$file $dest/$file
done
