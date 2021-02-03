""" Downstream network architectures"""
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from downstream.models.tdnnf import get_tdnnf_model


class TDNNFClassifier(nn.Module):
    """
        This class defines TDNNF architecture

        Args:
            input_dim: Dimension for the input features (int, required)
            output_dim: Dimension for the output (int, requred)
            config: configuration file containing tdnnf specific information
    """
    def __init__(self, input_dim, output_dim, config):
        super(TDNNFClassifier, self).__init__()

        self.output_subsampling = config['output_subsampling']
        self.pdrop = config['drop']
        self.size = config['size']
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.tdnnf = get_tdnnf_model(
            input_dim,
            output_dim,
            self.size,
            self.pdrop
        )

    def forward(self, features):
        chain_out, xent_out = self.tdnnf(features)
        return chain_out, xent_out
