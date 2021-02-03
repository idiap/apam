""" TDNNF architectures"""
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_value_

import pkwrap
from downstream.models.tdnnf_model import ChainModel


def initialize_model(input_dim, output_dim, **kwargs):
    model = ChainModel(input_dim, output_dim, **kwargs)
    return model


def get_tdnnf_model(input_dim, output_dim, size='small', dropout=0.1):
    p_dropout = dropout
    if size == 'small':
        padding = 7
        kernel_list = [3]*2 + [1] + [3]*2
        subsampling_list = [1]*2 + [3] + [1]*2

    if size == 'medium':
        padding = 11
        kernel_list = [3]*3 + [1] + [3]*3
        subsampling_list = [1]*3 + [3] + [1]*3

    if size == 'large':
        padding = 26
        kernel_list = [3]*3 + [1] + [3]*8
        subsampling_list = [1]*3 + [3] + [1]*8

    kwargs = {
        "padding": padding,
        "ivector_dim": 0,
        "hidden_dim": 1024,
        "bottleneck_dim": 128,
        "prefinal_bottleneck_dim": 256,
        "kernel_size_list": kernel_list,
        "subsampling_factor_list": subsampling_list,
        "frame_subsampling_factor": 3,
        "p_dropout": p_dropout
    }

    model = initialize_model(input_dim, output_dim, **kwargs)
    return model
