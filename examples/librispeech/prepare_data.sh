#!/bin/bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,


set -e
. ./cmd.sh
. ./path.sh

stage=0
train_set=train_clean_100
suffix=mam-large-tdnnf-med-ft-spec

# Update the corpus paths to your own
# corpus=./database/LibriSpeech
corpus=""
lm_url=www.openslr.org/resources/11

# option related to tree/phone-lm
shared_phones='true'
phones_type='biphone'

# Options related to egs generation for training
get_egs_stage=-10
cmvn_opts=""
left_context=0
right_context=0
frames_per_iter=3000000
frame_subsampling_factor=3
num_utts_subset=300
num_egs_diagnostic=300
num_train_egs_combine=10

. ./utils/parse_options.sh

if [ -z $corpus ]; then
    echo "Please set the corpus correctly"
    exit
fi

# Setting directory names
new_lang=data/lang_e2e_${phones_type}
treedir=exp/chain/e2e_${phones_type}_tree  # it's actually just a trivial tree (no tree building)
dir=exp/chain/e2e_${suffix}


if [ $stage -le -1 ]; then
  local/download_lm.sh $lm_url data/local/lm
fi

if [ $stage -le 0 ]; then
  # format the data as Kaldi data directories
  if [ ! -d "data/local/lm" ]; then
      echo "Exiting because data/local/lm folder doesn't exist"
      echo "Please set it up by either of two options:"
      echo "running from stage -1 OR copying/linking to existing folders"
      exit
  fi
  for part in test-clean dev-clean test-other dev-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $corpus/$part data/$(echo $part | sed s/-/_/g)
  done
fi

if [ $stage -le 1 ]; then

  mkdir -p data/local/lm_less_phones
  ln -rs data/local/lm/3-gram.arpa.gz data/local/lm_less_phones/lm_tglarge.arpa.gz
  ln -rs data/local/lm/3-gram.pruned.1e-7.arpa.gz data/local/lm_less_phones/lm_tgmed.arpa.gz
  ln -rs data/local/lm/3-gram.pruned.3e-7.arpa.gz data/local/lm_less_phones/lm_tgsmall.arpa.gz
  ln -rs data/local/lm/4-gram.arpa.gz data/local/lm_less_phones/lm_fglarge.arpa.gz
  cp data/local/lm/librispeech-vocab.txt data/local/lm_less_phones/
  cat data/local/lm/librispeech-lexicon.txt | sed -e 's/[0,1,2]//g' > \
    data/local/lm_less_phones/librispeech-lexicon.txt

  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$cpu_cmd" \
    data/local/lm_less_phones data/local/lm_less_phones data/local/dict_lp

  utils/prepare_lang.sh \
    --position_dependent_phones false \
    --share_silence_phones true \
    data/local/dict_lp \
    "<UNK>" data/local/lang_tmp_lp data/lang_lp

  local/format_lms.sh --src-dir data/lang_lp data/local/lm_less_phones
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm_less_phones/lm_tglarge.arpa.gz \
    data/lang_lp data/lang_lp_test_tglarge
  utils/build_const_arpa_lm.sh data/local/lm_less_phones/lm_fglarge.arpa.gz \
    data/lang_lp data/lang_lp_test_fglarge
fi

if [ $stage -le 2 ]; then
  for part in test_clean dev_clean test_other dev_other; do
    utils/copy_data_dir.sh data/$part data/${part}_hires
  done

  # utils/combine_data.sh data/train_960/ \
  #   data/train_clean_100 \
  #   data/train_clean_360 \
  #   data/train_other_500

  echo "$0: perturbing the training data to allowed lengths"
  # Uncomment the following command block if you need to firt exptract segments
  # For example in switchboard data preparation

  # utils/data/extract_wav_segments_data_dir.sh --nj 66 \
  #   --cmd "$cpu_cmd" data/$trainset data/${trainset}_wav
  # utils/data/get_utt2dur.sh --nj 66 --cmd "$cpu_cmd" \
  #   data/${trainset}_wav # necessary for the next command
  # # 12 in the following command means the allowed lengths are spaced
  # # by 12% change in length.
  # utils/data/perturb_speed_to_allowed_lengths.py 12 \
  #   data/${trainset}_wav \
  #   data/${trainset}_sp_hires

  # Comment out the following two commands if you uncomment the above block 
  utils/data/get_utt2dur.sh data/${train_set} # necessary for the next command
  # 12 in the following command means the allowed lengths are spaced
  # by 12% change in length.
  utils/data/perturb_speed_to_allowed_lengths.py 12 \
    data/${train_set} \
    data/${train_set}_sp_hires

  cat data/${train_set}_sp_hires/utt2dur | \
    awk '{print $1 " " substr($1,5)}' >data/${train_set}_sp_hires/utt2uniq
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires
  utils/fix_data_dir.sh data/${train_set}_sp_hires

  for part in test_clean dev_clean test_other dev_other ${train_set}_sp; do
    datadir=${part}_hires
    # Extracting 80 dim filter bank features
    mkdir -p data/feats/fbank
    steps/make_fbank.sh --fbank-config conf/fbank_hires.conf \
      --cmd "$cpu_cmd" --nj 50 data/${datadir} \
      data/feats/fbank/${datadir} data/feats/fbank/${datadir}/data || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir} \
      data/feats/fbank/${datadir} data/feats/fbank/${datadir}/data || exit 1;
    utils/fix_data_dir.sh data/${datadir} || exit 1
  done
fi

# feature extraction ends here
if [ $stage -le 3 ]; then
  bash shutil/chain/check_lang.sh data/lang_lp $new_lang
fi

if [ $stage -le 4 ]; then
  echo 'Estimating a phone language model for the denominator graph...'
  bash shutil/chain/estimate_e2e_phone_lm.sh --cmd "$cpu_cmd" \
    data/lang_lp $treedir \
    data/${train_set}_sp_hires $shared_phones $phones_type $new_lang
fi

if [ $stage -le 5 ]; then
  mkdir -p ${dir}/configs
  mkdir -p ${dir}/init
  cp -r $treedir/tree $dir/
  cp $treedir/phones.txt $dir/
  cp $treedir/phone_lm.fst $dir/
  cp $treedir/0.trans_mdl $dir/
  echo 'Making denominator fst for training'
  bash shutil/chain/make_e2e_den_fst.sh \
    --cmd "$cpu_cmd" $treedir $dir
fi

if [ $stage -le 6 ]; then
  echo 'Generating egs to be used during the training'
  bash steps/chain/e2e/get_egs_e2e.sh \
    --cmd "$cpu_cmd" \
    --cmvn-opts  "$cmvn_opts" \
    --left-context $left_context \
    --right-context $right_context \
    --frame-subsampling-factor $frame_subsampling_factor \
    --stage $get_egs_stage \
    --frames-per-iter $frames_per_iter \
    --num_utts_subset $num_utts_subset \
    --num_train_egs_combine $num_train_egs_combine \
    --num_egs_diagnostic $num_egs_diagnostic \
    --srand 1234 \
    data/${train_set}_sp_hires $dir $treedir $dir/egs
fi

if [ $stage -le 7 ]; then
  echo 'Dumping output units and feature dimensions for training'
  num_targets=$(tree-info ${treedir}/tree | grep num-pdfs | awk '{print $2}')
  echo $num_targets > $dir/num_pdfs
  cp $dir/egs/info/feat_dim $dir/feat_dim
fi

if [ $stage -le 8 ]; then
  echo 'Preparing the validation folder for tracking WER'
  mkdir -p $dir/egs/valid_diagnostic
  nnet3-chain-copy-egs \
    ark:$dir/egs/valid_diagnostic.cegs ark,t:- \
    | grep 'NumInputs' | sed -e 's/<\/Nnet3ChainEg> //g' |  cut -d ' ' -f 1 > $dir/egs/valid_diagnostic/utt_ids
  echo "dummy text" > $dir/egs/valid_diagnostic/text
  rm  $dir/egs/valid_diagnostic/text
  for id in `cat  $dir/egs/valid_diagnostic/utt_ids`; do
    grep $id data/${train_set}_sp_hires/text >>  $dir/egs/valid_diagnostic/text
  done
fi

# Generate a decoding graph to decode the validation data
# for early stopping
if [ $stage -le 9 ]; then
  cp $dir/0.trans_mdl $dir/final.mdl
  utils/lang/check_phones_compatible.sh \
    data/lang_lp_test_tgsmall/phones.txt $new_lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 --remove-oov data/lang_lp_test_tgsmall \
    $dir $treedir/graph_tgsmall || exit 1;
  rm $dir/final.mdl
fi
