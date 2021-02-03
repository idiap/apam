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
#   FileName     [ transformer/model.py ]
#   Synopsis     [ Implementation of the transformer models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/huggingface/transformers ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import sys
from collections import OrderedDict
import copy
import math
import numpy as np
import pprint
from io import open

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import LayerNorm as TransformerLayerNorm

from fast_transformers.masking import LengthMask
from fast_transformers.builders import TransformerEncoderBuilder


class TransformerConfig(object):
    """Configuration class to store the configuration of a `TransformerModel`.
    """
    def __init__(self, config):
        self.downsample_rate = config['transformer']['downsample_rate']
        self.hidden_size = config['transformer']['hidden_size']
        self.num_hidden_layers = config['transformer']['num_hidden_layers']
        self.num_attention_heads = config['transformer']['num_attention_heads']
        self.hidden_act = config['transformer']['hidden_act']
        self.intermediate_size = config['transformer']['intermediate_size']
        self.hidden_dropout_prob = config['transformer']['hidden_dropout_prob']
        self.attention_probs_dropout_prob = config['transformer']['attention_probs_dropout_prob']
        self.initializer_range = config['transformer']['initializer_range']
        self.layer_norm_eps = float(config['transformer']['layer_norm_eps'])

        self.attention_type = config['transformer']['attention_type'] if 'attention_type' in config['transformer'] else 'full'
        self.softmax_temp = float(config['transformer']['softmax_temp']) if 'softmax_temp' in config['transformer'] else None
        self.clusters = int(config['transformer']['clusters']) if 'clusters' in config['transformer'] else 200
        self.topk = int(config['transformer']['topk']) if 'topk' in config['transformer'] else 32
        self.bits = int(config['transformer']['bits']) if 'bits' in config['transformer'] else 63
        self.iterations = int(config['transformer']['iterations']) if 'iterations' in config['transformer'] else 10
        self.hash_bias = bool(config['transformer']['hash_bias']) if 'hash_bias' in config['transformer'] else True
        self.share_layer = bool(config['transformer']['share_layer']) if 'share_layer' in config['transformer'] else False
        self.pre_layer_norm = bool(config['transformer']['pre_layer_norm']) if 'pre_layer_norm' in config['transformer'] else False
        self.local_context = int(config['transformer']['context']) if 'context' in config['transformer'] else 150
        self.length_limit = int(config['transformer']['length_limit']) if 'length_limit' in config['transformer'] else 512
        #pprint.pprint(config['transformer'])


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class TransformerInputRepresentations(nn.Module):
    """Construct the input representation from spectrogram, and position encodings.
    """
    def __init__(self, config, input_dim):
        super(TransformerInputRepresentations, self).__init__()
        self.hidden_size = config.hidden_size
        self.spec_transform = nn.Linear(input_dim * config.downsample_rate, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, spec, pos_enc):
        spec_transformed = self.spec_transform(spec)

        input_representations = spec_transformed + pos_enc
        input_representations = self.LayerNorm(input_representations)
        input_representations = self.dropout(input_representations)
        return input_representations

class TransformerEncoder(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(TransformerEncoder, self).__init__()
        self.output_attentions = output_attentions
        self.pre_layer_norm = config.pre_layer_norm
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            feed_forward_dimensions=config.intermediate_size,
            query_dimensions=int(config.hidden_size / config.num_attention_heads),
            value_dimensions=int(config.hidden_size / config.num_attention_heads),
            dropout=config.hidden_dropout_prob
        )
        if config.softmax_temp:
            builder.attention.softmax_temp = config.softmax_temp
        builder.attention.attention_dropout = config.attention_probs_dropout_prob
        builder.attention.clusters = config.clusters
        builder.attention.bits = config.bits
        builder.attention.hash_bias = config.hash_bias
        builder.attention.iterations = config.iterations
        builder.attention.topk = config.topk
        builder.attention.local_context = config.local_context
        builder.attention.length_limit = config.length_limit
        attention_type = config.attention_type
        if attention_type == "improved-clustered":
            attention_type = "conditional-full:improved-clustered"
        builder.attention_type = attention_type
        self.transformer = builder.get()

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False, head_mask=None):
        input_lengths = attention_mask.sum(-1)
        encoded_output = self.transformer(
            hidden_states, attn_mask=None,
            length_mask=LengthMask(
                lengths=input_lengths.long(),
                max_len=int(max(max(input_lengths),hidden_states.shape[1]))
            )
        )
        return encoded_output


class TransformerSpecPredictionHead(nn.Module):
    def __init__(self, config, output_dim, input_dim=None):
        super(TransformerSpecPredictionHead, self).__init__()
        self.output_dim = output_dim
        if input_dim is None:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(input_dim, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.hidden_size, self.output_dim * config.downsample_rate)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        linear_output = self.output(hidden_states)
        return linear_output, hidden_states


class TransformerInitModel(nn.Module):
    """ An abstract class to handle weights initialization."""
    def __init__(self, config, output_attentions, *inputs, **kwargs):
        super(TransformerInitModel, self).__init__()
        self.config = config
        self.output_attentions = output_attentions

    def init_Transformer_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, TransformerLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class TransformerModel(TransformerInitModel):
    """Transformer model.

    Params:
        `config`: a TransformerConfig class instance with the configuration to build a new model
        `intput_dim`: int,  input dimension of model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
    Inputs:
        `spec_input`: a torch.LongTensor of shape [batch_size, sequence_length, feature_dimension]
            with the selected frames processed as masked frames during training,
            generated by the `process_MAM_data()` function in `solver.py`.
        `pos_enc`: a torch.LongTensor of shape [batch_size, sequence_length, hidden_size],
            generated by the `position_encoding()` function in `solver.py`.
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.


    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states
                at the end of each attention block, each encoded-hidden-state is a torch.FloatTensor
                of size [batch_size, sequence_length, hidden_size], i.e [num_hidden_layers, batch_size, sequence_length, hidden_size]
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size].


    Example usage:
    ```python
    spec_input = torch.LongTensor(spec_frames)
    pos_enc = torch.LongTensor(position_encoding(seq_len=len(spec_frames)))

    config = TransformerConfig(hidden_size=768,
             num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = TransformerForMaskedLM(config)
    masked_spec_logits = model(spec_input, pos_enc)
    ```
    """
    def __init__(self, config, input_dim, output_attentions=False, keep_multihead_output=False, with_input_module=True):
        super(TransformerModel, self).__init__(config, output_attentions)
        self.with_input_module = with_input_module
        if self.with_input_module: self.input_representations = TransformerInputRepresentations(config, input_dim)
        self.encoder = TransformerEncoder(config, output_attentions=output_attentions,
                                          keep_multihead_output=keep_multihead_output)
        self.apply(self.init_Transformer_weights)

    def forward(self, spec_input, pos_enc=None, attention_mask=None, output_all_encoded_layers=False, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(spec_input)

        if self.with_input_module:
            input_representations = self.input_representations(spec_input, pos_enc)
        else:
            input_representations = spec_input
        encoded_output = self.encoder(input_representations,
                                      attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask)
        return encoded_output


class TransformerForMaskedAcousticModel(TransformerInitModel):
    """Transformer model with the masked acoustic modeling head.
    This module comprises the Transformer model followed by the masked acoustic modeling head.

    Params:
        `config`: a TransformerConfig class instance with the configuration to build a new model
        `intput_dim`: int,  input dimension of model
        `output_dim`: int,  output dimension of model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `spec_input`: a torch.LongTensor of shape [batch_size, sequence_length, feature_dimension]
            with the selected frames processed as masked frames during training,
            generated by the `process_MAM_data()` function in `solver.py`.
        `pos_enc`: a torch.LongTensor of shape [batch_size, sequence_length, hidden_size],
            generated by the `position_encoding()` function in `solver.py`.
        `masked_label`: masked acoustic modeling labels - torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [1, 0]. All labels set to -1 are ignored, the loss
            is only computed for the labels set to 1.
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `spce_label`: a torch.LongTensor of shape [batch_size, sequence_length, feature_dimension]
            which are the ground truth spectrogram used as reconstruction labels.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.

    Outputs:
        if `spec_label` and `mask_label` is not `None`:
            Outputs the masked acoustic modeling loss and predicted spectrogram.
        if `spec_label` and `mask_label` is `None`:
            Outputs the masked acoustic modeling predicted spectrogram of shape [batch_size, sequence_length, output_dim * downsample_rate].

    Example usage:
    ```python
    spec_input = torch.LongTensor(spec_frames)
    pos_enc = torch.LongTensor(position_encoding(seq_len=len(spec_frames)))

    config = TransformerConfig(hidden_size=768,
             num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = TransformerForMaskedLM(config)
    masked_spec_logits = model(spec_input, pos_enc)
    ```
    """
    def __init__(self, config, input_dim, output_dim,
                 output_attentions=False, keep_multihead_output=False):
        super(TransformerForMaskedAcousticModel, self).__init__(
            config, output_attentions
        )
        self.Transformer = TransformerModel(
            config, input_dim,
            output_attentions=output_attentions,
            keep_multihead_output=keep_multihead_output
        )
        self.SpecHead = TransformerSpecPredictionHead(
            config,
            output_dim if output_dim is not None else input_dim
        )
        self.apply(self.init_Transformer_weights)
        self.loss = nn.L1Loss()

    def forward(self, spec_input, pos_enc,
                mask_label=None, attention_mask=None,
                spec_label=None, head_mask=None):
        sequence_output = self.Transformer(
            spec_input, pos_enc, attention_mask,
            output_all_encoded_layers=False,
            head_mask=head_mask
        )
        pred_spec, pred_state = self.SpecHead(sequence_output)

        if spec_label is not None and mask_label is not None:
            assert mask_label.sum() > 0, 'Without any masking, loss might go NaN. Modify your data preprocessing (utility/mam.py)'
            masked_spec_loss = self.loss(pred_spec.masked_select(mask_label),
                                         spec_label.masked_select(mask_label))
            return masked_spec_loss, pred_spec
        return pred_spec, pred_state, sequence_output
