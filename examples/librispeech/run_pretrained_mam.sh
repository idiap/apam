#!/bin/bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,


set -e
. ./cmd.sh
. ./path.sh

echo "$0 $@"

stage=0
suffix=mam-large-tdnnf-med-ft-spec

# Options related to acoustic model training
train_stage=0
cmvn_opts=""

ckpt=''

downconfig=conf/mam/asr-tdnnf-med-downstream-6h.yaml
upconfig=conf/mam/upstream_clustered.yaml
. ./utils/parse_options.sh

if [ -z $ckpt ]; then
    echo "Please set the ckpt correctly"
    echo "ckpt should contain the path to the pretrained model"
    exit
fi

# Setting directory names
treedir=exp/chain/e2e_biphone_tree  # it's actually just a trivial tree (no tree building)

dir=exp/chain/e2e_$suffix

echo $dir

transformer_model_file='src/e2e/mam_model.py'
# Initialize the model
if [ $stage -le 0 ]; then
  $gpu_cmd \
    $dir/log/init.log \
    $transformer_model_file \
    --mode 'init' \
    --exp_dir $dir \
    --config $downconfig \
    --upconfig $upconfig \
    --ckpt $ckpt
fi


# Train the model
if [ $stage -le 1 ]; then
  python src/e2e/train.py \
    --model $transformer_model_file \
    --exp_dir $dir \
    --train_stage $train_stage \
    --config $downconfig \
    --upconfig $upconfig \
    --ckpt $ckpt \
    --graph_dir $treedir/graph_tgsmall
fi
