#!/usr/bin/env python3
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>

import argparse
import os
import sys
import json
from typing import NamedTuple
import math
from collections import OrderedDict

import numpy as np
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn.utils import clip_grad_value_

import pkwrap

from parsers.exp_config import add_exp_options, \
    print_exp_options
from parsers.downstream_config import add_downstream_options, \
    print_downstream_options

from lfmmi.runner import LFMMITrainer
from pretrained.mam import initialize_mam_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    # General arguments
    parser.add_argument("--mode", default="init")
    parser.add_argument("--new_model", default="", type=str)
    parser.add_argument("--merge_models", default="", type=str)

    # Training related arguments
    parser.add_argument("--lr_downstream", type=float)

    # Egs related argument
    parser.add_argument("--cegs_indx", default="", type=str)

    # Loss tracking related arguments
    parser.add_argument("--validation_model", default="", type=str)
    parser.add_argument("--loss_validation", default='False', choices=['True', 'False'], type=str)
    parser.add_argument("--diagnostic", default='valid', type=str)
    parser.add_argument("--decode_validation", default='False', choices=['True', 'False'], type=str)

    # Decoding related arguments
    parser.add_argument("--decode_feats", default="data/test/feats.scp", type=str)
    parser.add_argument("--decode_output", default="-", type=str)
    parser.add_argument("--decode_gpu", default='False', choices=['True', 'False'], type=str)

    parser = add_downstream_options(parser)
    parser = add_exp_options(parser)

    args = parser.parse_args()

    runner = LFMMITrainer(
        args.exp_dir,
        args.config, args.upconfig,
        args.ckpt, initialize_mam_model
    )

    args.decode_gpu = strtobool(args.decode_gpu)
    args.decode_validation = strtobool(args.decode_validation)
    args.loss_validation = strtobool(args.loss_validation)

    if args.mode == 'init':
        upstream, downstream = runner.initialize_model()
        runner.save_model(
            upstream, downstream, None, os.path.join(args.exp_dir, "0.pt")
        )

    elif args.mode == 'training':
        runner.run_train(
            args.train_stage, args.lr_downstream,
            args.cegs_indx, args.new_model
        )

    elif args.mode == 'validation':
        runner.run_validation(
            args.validation_model, args.diagnostic,
            args.loss_validation, args.decode_validation,
            args.decode_gpu, args.decode_output
        )

    elif args.mode == 'decode':
        runner.decode(
            args.train_stage, args.decode_feats,
            args.decode_output, args.decode_gpu
        )

    elif args.mode == 'merge':
        runner.merge(args.merge_models, args.new_model)

    else:
        sys.stderr.write('Choose one of the available modes')
