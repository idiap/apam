#!/bin/bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,


set -e

. ./cmd.sh
. ./path.sh

stage=1
model_iter="75"
suffix=mam-large-tdnnf-med-ft-spec

cmvn_opts=""
data="dev_clean"

# We use vanilla attention for decoding
# Slightly better WER
upconfig=conf/mam/upstream_full.yaml
downconfig=conf/mam/asr-tdnnf-med-downstream-6h.yaml
ckpt=""

. ./utils/parse_options.sh

if [ -z $ckpt ]; then
    echo "Please set the ckpt correctly"
    echo "ckpt should contain the path to the pretrained model"
    exit
fi

dir="exp/chain/e2e_${suffix}/"
treedir="exp/chain/e2e_biphone_tree"
graph_dir="${treedir}/graph_tgsmall"

data_dir="data/${data}_hires"
decode_dir="$dir/decode_${data}_tgsmall/${model_iter}"

# Split decoding data
nspk=$(wc -l <$data_dir/spk2utt)
nj_max=40
nj=$(( nspk > nj_max ? nj_max : nspk ))
sdata="$data_dir/split$nj"
[[ -d $sdata && $data_dir/feats.scp -ot $sdata ]] || split_data.sh $data_dir $nj || exit 1;

transformer_model_file="src/e2e/mam_model.py"
job_fol="${data_dir}/split$nj/JOB"
decode_feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$job_fol/utt2spk scp:$job_fol/cmvn.scp scp:$job_fol/feats.scp ark:- |"
if [ $stage -le 1 ]; then
  $cpu_cmd JOB=1:$nj $decode_dir/log/decode.JOB.log \
    $transformer_model_file \
      --mode decode \
      --exp_dir $dir \
      --train_stage "$model_iter" \
      --config $downconfig \
      --upconfig $upconfig \
      --ckpt $ckpt \
      --decode_feat  "$decode_feats" \
      "|" shutil/decode/latgen-faster-mapped.sh \
      $graph_dir/words.txt \
      $dir/0.trans_mdl $graph_dir/HCLG.fst $decode_dir/lat.JOB.gz
fi

if [ $stage -le 2 ]; then
  cp $dir/0.trans_mdl $decode_dir/../final.mdl
  local/score.sh --cmd "$cpu_cmd" \
    --min_lmwt 6 --max_lmwt 17 \
    $data_dir $graph_dir $decode_dir
fi

if [ $stage -le 3 ]; then
  grep WER $decode_dir/wer_* | utils/best_wer.sh
fi

rescore_dir="$dir/decode_${data}_fglarge/${model_iter}"
if [ $stage -le 4 ]; then
  numjobs=`ls ${decode_dir} | grep lat | wc -l`
  echo $numjobs > ${decode_dir}/num_jobs
  mkdir -p $rescore_dir
  cp ${decode_dir}/../final.mdl ${rescore_dir}/../final.mdl
  steps/lmrescore_const_arpa.sh --cmd "${decode_cmd}" \
    data/lang_lp_test_{tgsmall,fglarge} \
    ${data_dir} ${decode_dir} ${rescore_dir}
fi

if [ $stage -le 5 ]; then
  grep WER $rescore_dir/wer_* | utils/best_wer.sh
fi
