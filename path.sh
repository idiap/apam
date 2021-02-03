# Environment variables for running Kaldi scripts
if [ -z $KALDI_ROOT ]; then
    export KALDI_ROOT=`pwd`/../../..
fi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$PWD:$PATH
for binfolder in bin chainbin featbin fgmmbin gmmbin fstbin ivectorbin latbin lmbin nnet2bin nnet3bin online2bin nnetbin onlinebin rnnlmbin; do
    export PATH=$KALDI_ROOT/src/$binfolder/:$PATH
done

export LC_ALL=C
# For now, don't include any of the optional dependenices of the main
# librispeech recipe
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KALDI_ROOT/src/lib/
