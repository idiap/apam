Please set the ```KALDI_ROOT``` correctly
and update the ```cmd.sh``` and ```config``` according to
correctly submit to the correct GPU and CPU queues.

Download the pretrained MAM model from [here](https://zenodo.org/record/4541045#.YCpThmgzaiw).

Follow the recipes in mam.md to finetune a model pretrained with MAM.

Note: Please update the corpus paths in data preparation scripts
Note: Please pay attention to the ckpt paths in the recipes. They should
point to the downloaded pretrained model.
