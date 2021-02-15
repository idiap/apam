# APAM - Adaptation of Pretrained Acoustic Models

APAM toolkit is built on PyTorch and provides recipes to adapt pretrained
acoustic models with a variety of sequence discriminative training criterions.

------------------------------------

Table of Contents
------------------------------------

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Introduction](#introduction)
   * [High-Level Library Structure](#high-level-library-structure)
   * [Installation](#installation)
       * [Dependencies](#dependencies)
   * [Pretrained Models Supported](#pretrained-models-supported)
       * [Masked Acoustic Model](#masked-acoustic-model)
   * [Current Recipes](#current-recipes)
       * [Librispeech 100h](#librispeech-100h)
   * [References](#references)
   * [Citation](#citation)
<!--te-->


------------------------------------
Introduction
------------------------------------
The library structure is inspired from the [S3PRL
library](https://github.com/s3prl/s3prl/). In keeping up with the terminology
in S3PRL, the pretrained models are referred to as *upstream* models. A
separate *downstream* model is added on the top of *upstream* model to be used
as acoustic model for ASR training. 


------------------------------------
High-Level Library Structure
------------------------------------
The library provides various runners (trainers) that take care of training
acoustic models.

The runner takes as input:

*asr_config*: which defines parameters related to experiment such as learning
rate, optimizers, epochs etc. 
*ckpt*: path to pretrained model ckpt 
*upconfig* configuration related to pretrained *upstream* model.
*get_model* function which create the *upstream* and *downstream* model using
the above parameters


The idea is to re-use the various pretrained models such as TERA, wav2vec 
through decoupled *upstream* and *downstream* models. This is enabled by writing
simple scripts to load these pretrained models. Examples for these can be found 
in *pretrained* folder in the source code.


------------------------------------
Installation
------------------------------------

### Dependencies
- **Python** 3 or above
- Required packages and their use are listed below:
```
torch                        # deep neural networks
pytorch-fast-transformers    # fast clustered attention
pkwrap                       # lfmmi loss
librosa                      # audio file reading
yaml                         # config parser
```

We recommend installing the latest version of [fast
transformers](https://github.com/idiap/fast-transformers) using the following
command:
```
pip install git+https://github.com/idiap/fast-transformers
```

To install Pkwrap follow the instructions here
[Pkwrap](https://github.com/idiap/pkwrap)


------------------------------------
Pretrained Models Supported
------------------------------------

At the moment we support the following pretrained models 

### Masked Acoustic Model

We provide the pretrained model for trained with masked language modeling
objective as described in ["TERA: Self-Supervised Learning of Transformer
Encoder Representation for Speech"](https://arxiv.org/abs/2007.06028).

The pretrained model is available [here](https://zenodo.org/record/4541045#.YCpThmgzaiw).

------------------------------------
Current Recipes
------------------------------------

At the moment, we only support [flat-start lattice-free
MMI](https://www.danielpovey.com/files/2018_interspeech_end2end.pdf) training.
The following recipes can be found in the examples folder. For more details
on how to run, follow the steps in the README files in examples

### Librispeech 100h
We provide recipes to train acoustic model using 100 hours of 
librispeech data and pretrained acoustic models based on 

1. Masked Acoustic Model


------------------------------------
Citation
------------------------------------
If you found this library useful, please cite the relevant work(s) from below
```bibtex
@misc{vyas2020latticefree,
    title={Lattice-Free MMI Adaptation Of Self-Supervised Pretrained Acoustic Models}, 
    author={Apoorv Vyas and Srikanth Madikeri and Hervé Bourlard},
    year={2020},
    eprint={2012.14252},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

------------------------------------
References
------------------------------------

Please note that this list is not exhaustive. We are only providing
references to a few key works which this library uses. For a more exhaustive list
please take a look at our published reports based on this library.


```bibtex
@inproceedings{paszke2019pytorch,
    title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
    author = {Paszke, Adam et. al.},
    booktitle = {Advances in Neural Information Processing Systems 32},
    year = {2019},
}
```

```bibtex
@article{hadian2018flat,
    author={Hossein Hadian and others},
    title={Flat-Start Single-Stage Discriminatively Trained HMM-Based Models for ASR},
    year={2018},
    journal={IEEE ACM Transactions on Audio, Speech, and Language Processing},
}
```

```bibtex
@misc{madikeri2020pkwrap,
    title={Pkwrap: a PyTorch Package for LF-MMI Training of Acoustic Models}, 
    author={Srikanth Madikeri and Sibo Tong and Juan Zuluaga-Gomez and Apoorv Vyas and Petr Motlicek and Herv{\'e} Bourlard},
    year={2020},
    eprint={2010.03466},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

```bibtex
@inproceedings{vyas2020fast,
    author = {Vyas, Apoorv and Katharopoulos, Angelos and Fleuret, Fran\c{c}ois},
    title = {Fast Transformers with Clustered Attention},
    booktitle = {Proceedings of the international conference on Neural Information Processing Systems (NeurIPS)},
    year = {2020}
}
```

```bibtex
@misc{
    S3PRL,
    author = {Andy T. Liu and Yang Shu-wen},
    title = {S3PRL: The Self-Supervised Speech Pre-training and Representation Learning Toolkit},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/s3prl/s3prl}
}
```
