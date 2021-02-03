# -*- coding: utf-8 -*- #

"""*********************************************************************************************"""
#   Author       [ Apoorv Vyas ]
"""*********************************************************************************************"""
# The following code is modified from the
# Self-Supervised Speech Pre-training and Representation Learning Toolkit
# provided here:
# https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning
# Original information is available below:
"""*********************************************************************************************"""
#   FileName     [ run_downstream.py ]
#   Synopsis     [ scripts for running downstream evaluation of upstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import yaml
import torch
import random
import argparse
import numpy as np
from shutil import copyfile
from torch.optim import Adam

from distutils.util import strtobool 
from transformer.nn_transformer import TRANSFORMER
from downstream.model import TDNNFClassifier


################
# GET UPSTREAM #
################
"""
Upstream model should meet the requirement of:
    1) Implement the `forward` method of `nn.Module`,
    2) Contains the `out_dim` attribute.
    3) Takes input and output in the shape of: (batch_size, time_steps, feature_dim)
"""
def get_upstream_model(upstream_opts, upconfig, input_dim, ckpt):
    start_new = strtobool(str(upstream_opts['start_new']))
    fine_tune = strtobool(str(upstream_opts['fine_tune']))
    specaug = strtobool(str(upstream_opts['specaug']))
    encoder_feat = strtobool(str(upstream_opts['encoder_feat']))
    options = {'ckpt_file'     : ckpt,
               'load_pretrain' : 'True' if not start_new else 'False',
               'no_grad'       : 'True' if not fine_tune else 'False',
               'dropout'       : 'default',
               'spec_aug'      : 'False',
               'spec_aug_prev' : 'True' if specaug else 'False',
               'weighted_sum'  : 'False',
               'select_layer'  : -1,
               'encoder_feat'  : 'True' if encoder_feat else 'False'
    }
    if upconfig == 'default':
        upconfig = None
    upstream_model = TRANSFORMER(options, input_dim, config=upconfig)
    upstream_model.permute_input = False
    assert(hasattr(upstream_model, 'forward'))
    assert(hasattr(upstream_model, 'out_dim'))
    return upstream_model


##################
# GET DOWNSTREAM #
##################
def get_downstream_model(input_dim, class_num, config):
    model_config = config['model']['downstream']
    model_name = str(model_config['architecture'])
    if model_name == 'tdnnf':
        downstream_model = TDNNFClassifier(input_dim, class_num, model_config)
    
    return downstream_model


def initialize_mam_model(
    asr_config_path, upconfig,
    input_dim, output_dim, ckpt
):
    config = yaml.load(open(asr_config_path, 'r'), Loader=yaml.FullLoader)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    upstream_opts = config['model']['upstream']
    upstream_model = get_upstream_model(upstream_opts, upconfig, input_dim, ckpt)
    downstream_model = get_downstream_model(
        upstream_model.out_dim,
        output_dim,
        config
    )
    return upstream_model, downstream_model
