## Steps to run
Update cmd.sh and conf files to set the submission command to the queues.

Update the paths in ```init_kaldi.sh```

Run the following to setup up the directories.
```
bash init_kaldi.sh
```

# Data Preparation
## Feature preparation
Update the paths in the ```prepare_data.sh```

Note 1: data and exp directory will require a lot
of space so point them to somewhere with large diskspace
Checkout the code to choose the right stage

Note 2: If you are using some old setup with pre-existing
features, make sure they were extracted with ```conf/fbank_hires.conf```
Otherwise, it doesn't match the pre-trained model
```
bash prepare_data.sh --stage -1 \
  --suffix mam-tdnnf-med-spec-ft \
```

## Helpful Trick for doing additional experiments (Optional)
For any additional experiments after dataprep, you can 
use the following command

```
bash misc/setup_next_exp.sh \
  <path_to_prev_exp> \
  <path_to_new_exp>
```

Example:
```
bash misc/setup_next_exp.sh \
  exp/e2e_mam-tdnnf-med-spec-ft \
  exp/e2e_mam-exp-2
```
## Model Training
## Finetuning on mam model
Please set the parameters in the experiment configuration  
file: "asr-tdnnf-med-downstream-6h.yaml"
Make sure that path to the pre-trained model is passed correctly
```
bash run_pretrained_mam.sh \
  --suffix mam-tdnnf-med-spec-ft \
  --downconfig conf/mam/asr-tdnnf-med-downstream-6h.yaml \
  --ckpt pretrained-mam/states-200000.ckpt \
  --stage 0 \
  --train-stage 0
```

If training crashes at any point,say iteration 5, we can pass run the same
command again but simply pass:
```
  --stage 1 \
  --train-stage 5
```

## Decoding
To decode we can run the following command
```
bash decode/decode_mam.sh \
  --suffix  mam-tdnnf-med-spec-ft \
  --model-iter 90 \
  --data dev_clean \
  --ckpt pretrained-mam/states-200000.ckpt
```
```model_iter``` is the model iteration we want to use for decoding
```data``` is the data porion we want to decode
Note that you may want to try multiple iterations on development data to choose
the best iteration for decoding.

Additionally, the model was trained with clustered attention to save memory,
at test time, we can use full attention to get a small boost in performance.
This can be done by additionally passing the upconfig as follows:
```
bash decode/decode_mam.sh \
  --suffix  mam-tdnnf-med-spec-ft \
  --model-iter 50 \
  --data dev_clean \
  --upconfig conf/mam/upstream_full.yaml \
  --ckpt pretrained-mam/states-200000.ckpt
```
