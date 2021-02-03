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
#   FileName     [ transformer/nn_transformer.py ]
#   Synopsis     [ wrapper class for downstream feature extraction or finetune ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import sys
import math
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from functools import lru_cache
from distutils.util import strtobool
from transformer.model import TransformerConfig, TransformerModel, \
        TransformerForMaskedAcousticModel


###############
# TRANSFORMER #
###############

class TRANSFORMER(nn.Module):
    """
    Use this class to extract features from the Transformer model,
    or to finetune the pre-trained Transformer with ASR.

    Arguments:
        `options`: a python dictionary containing the following keys:
            ckpt_file: str, a path specifying the pre-trained ckpt file
            load_pretrain: str, ['True', 'False'], whether to load pre-trained weights
            no_grad: str, ['True', 'False'], whether to have gradient flow over this class
            dropout: float/str, use float to modify dropout value during downstream finetune, or use the str `default` for pre-train default values
            spec_aug: str, ['True', 'False'], whether to apply SpecAugment on inputs (used for ASR training)
            spec_aug_prev: str, ['True', 'False'], apply spec augment on input acoustic features if True, else apply on output representations (used for ASR training)
            encoder_feat: str, ['True', 'False'], if True, use encoder features otherwise use last hidden layer before reconstruction
            weighted_sum: str, ['True', 'False'], whether to use a learnable weighted sum to integrate hidden representations from all layers, if False then use the last
            select_layer: int, select from all hidden representations, set to -1 to select the last (will only be used when weighted_sum is False)
        `input_dim`: int, input dimension of model

    An example `options` dictionary:
    options = {
        'ckpt_file'     : './result/result_transformer/libri_sd1337_fmllrBase960-F-N-K-RA/states-1000000.ckpt',
        'load_pretrain' : 'True',
        'no_grad'       : 'True',
        'dropout'       : 'default',
        'spec_aug'      : 'False',
        'spec_aug_prev' : 'True',
        'encoder_feat'  : 'True',
        'weighted_sum'  : 'False',
        'select_layer'  : -1,
    }
    """
    def __init__(self, options, inp_dim, config=None):
        super(TRANSFORMER, self).__init__()

        all_states = torch.load(options["ckpt_file"], map_location='cpu')
        self.config = all_states['Settings']['Config']

        if config is not None:
            self.config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
        self.no_grad = bool(strtobool(options['no_grad']))
        self.spec_aug = bool(strtobool(options['spec_aug']))
        self.spec_aug_prev = bool(strtobool(options['spec_aug_prev']))
        self.weighted_sum = bool(strtobool(options['weighted_sum']))
        self.select_layer = int(options['select_layer'])

        # increase dropout
        if str(options['dropout']) != 'default':
            self.config['transformer']['hidden_dropout_prob'] = \
                    float(options['dropout'])
            self.config['transformer']['attention_probs_dropout_prob'] = \
                    float(options['dropout'])

        # Model Config
        self.model_config = TransformerConfig(self.config)
        self.dr = self.model_config.downsample_rate
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.max_input_length = self.config['transformer']['max_input_length'] if 'max_input_length' in self.config['transformer'] else 0
        if self.max_input_length > 0: sys.stderr.write('[Transformer] - Maximum input length: {}'.format(self.max_input_length))

        if not (self.select_layer in list(range(-1, self.num_layers))):
            raise RuntimeError('Out of range int for \'select_layer\'!')

        # use weighted sum from all layers
        if self.weighted_sum:
            self.weight = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)

        # Build model
        self.device = torch.device('cpu')
        self.encoder_feat = bool(strtobool(options['encoder_feat']))
        self.model = TransformerForMaskedAcousticModel(
            self.model_config, inp_dim, None
        ).to(self.device)
        self.model.eval() if self.no_grad else self.model.train()

        # Load from a PyTorch state_dict
        load = bool(strtobool(options["load_pretrain"]))
        if load:
            self.load_spechead_model(all_states)
            sys.stderr.write(
                '[Transformer] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.out_dim = self.hidden_size
        self.permute_input = True # This attribute is for the forward method. If True then input ouput is in the shape of (T, B, D), if False then in (B, T, D)

    def set_device(self, device='cpu'):
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                sys.stderr.write('No cuda found')
        self.model = self.model.to(self.device)
        return

    def load_spechead_model(self, all_states):
        try:
            self.model.SpecHead.load_state_dict(all_states['SpecHead'])
            sys.stderr.write('[SpecHead] - Loaded')
        except: sys.stderr.write('[SpecHead - X]')

        try:
            state_dict = all_states['Transformer']
            self.model.Transformer.load_state_dict(state_dict)
            sys.stderr.write('[Transformer] - Loaded')
        except: sys.stderr.write('[Transformer - X]')


    def load_model(self, all_states):
            try:
                # perform load
                state_dict = all_states['Transformer']
                self.model.load_state_dict(state_dict)
                sys.stderr.write('[Transformer] - Loaded')
            except: sys.stderr.write('[Transformer - X]')

    def up_sample_frames(self, spec, return_first=False):
        if len(spec.shape) != 3:
            spec = spec.unsqueeze(0)
            assert(len(spec.shape) == 3), 'Input should have acoustic feature of shape BxTxD'
        # spec shape: [batch_size, sequence_length // downsample_rate, output_dim * downsample_rate]
        spec_flatten = spec.view(spec.shape[0], spec.shape[1]*self.dr, spec.shape[2]//self.dr)
        if return_first: return spec_flatten[0]
        return spec_flatten # spec_flatten shape: [batch_size, sequence_length * downsample_rate, output_dim // downsample_rate]

    def down_sample_frames(self, spec):
        spec = spec.contiguous()
        left_over = spec.shape[1] % self.dr
        if left_over != 0: spec = spec[:, :-left_over, :]
        spec_stacked = spec.view(spec.shape[0], spec.shape[1]//self.dr, spec.shape[2]*self.dr)
        return spec_stacked


    def process_input_data(self, spec):
        """Process input data for the model"""
        # add arbitary batch axis B if input `spec` has shape of TxD

        if len(spec.shape) == 2:
            spec = spec.unsqueeze(0)
        # input `spec` should have shape BxTxD
        elif len(spec.shape) != 3:
            raise ValueError('Input argument `spec` has invalid shape: {}'.format(spec.shape))

        # Down sample
        if self.dr > 1:
            spec_stacked = self.down_sample_frames(spec) # (batch_size, seq_len, feature_dim * dr)
        else:
            spec_stacked = spec

        # Record length for each uttr
        spec_len = np.sum(np.sum(spec_stacked.cpu().data.numpy(), axis=-1) != 0, axis=-1)
        spec_len = [int(sl) for sl in spec_len]

        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]

        pos_enc = position_encoding(seq_len, self.hidden_size) # (seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(spec_stacked)):
            attn_mask[idx][spec_len[idx]:] = 0
        if self.spec_aug_prev and self.model.training:
            spec_stacked = spec_augment(spec_stacked, mask_T=70, mask_F=4, num_T=2, num_F=2, p=1.0) # (batch_size, seq_len, feature_dim * dr)
        spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32) # (batch_size, seq_len, feature_dim * dr)
        pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32).expand(spec_stacked.size(0), *pos_enc.size()) # (batch_size, seq_len, hidden_size)
        attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32) # (batch_size, seq_len)
        return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)


    def tile_representations(self, reps):
        """
        Tile up the speech representations to match the amount of input frames.
        Input - encoded_layers shape: (batch_size, sequence_length, hidden_size)
        Output - tiled_encoded_layers shape: (batch_size, sequence_length * downsample_rate, hidden_size)
        """
        if len(reps.shape) != 3:
            raise ValueError('Input argument `reps` has invalid shape: {}'.format(reps.shape))

        tiled_reps = reps.repeat(1, 1, self.dr)
        tiled_reps = tiled_reps.reshape(reps.size(0), reps.size(1)*self.dr, reps.size(2))
        return tiled_reps # (batch_size, sequence_length * downsample_rate, hidden_size)


    def upsample(self, x, input_len):
        # Compute padding to compromise the downsample loss
        left_over = input_len % self.dr
        if left_over % 2 == 0:
            left_pad = left_over // 2
            right_pad = left_pad
        else:
            left_pad = left_over // 2
            right_pad = left_over // 2 + 1

        x = self.tile_representations(x)

        # padding
        x = x.permute(0, 2, 1).contiguous() # (B, T, D) -> (B, D, T)
        padding = nn.ReplicationPad1d((left_pad, right_pad))
        x = padding(x)

        x = x.permute(0, 2, 1).contiguous() # (B, D, T) -> (B, T, D)
        return x


    def _forward(self, x):

        if self.permute_input:
            x = x.permute(1, 0, 2).contiguous() # (T, B, D) -> (B, T, D)
        input_len = x.shape[1]

        # forward the whole sequence at once
        if self.max_input_length == 0 or input_len <= self.max_input_length:
            spec_stacked, pos_enc, attn_mask = self.process_input_data(x) # x shape: (B, T, D)
            pred_spec, pred_state, enc_state = self.model(spec_stacked, pos_enc, None, attn_mask, None, None) # (B, T, D) or # (N, B, T, D)
            if self.encoder_feat:
                x = enc_state
            else:
                x = pred_state
        # forward the sequence in chunks then concat
        else:
            chunks = torch.chunk(x, chunks=math.ceil(input_len / self.max_input_length), dim=1)
            x_ = []
            for chunk in chunks:
                spec_stacked, pos_enc, attn_mask = self.process_input_data(chunk) # x shape: (B, T, D)
                pred_spec, pred_state, enc_state = self.model(spec_stacked, pos_enc, None, attn_mask, None, None) # (B, T, D) or # (N, B, T, D)
                if self.encoder_feat:
                    chunk = enc_state
                else:
                    chunk = pred_state
                x_.append(torch.stack(chunk) if type(chunk) is list else chunk)
            x = torch.cat(x_, dim=2 if (self.weighted_sum or self.select_layer != -1) else 1)

        # Apply weighted sum
        if self.weighted_sum:
            if type(x) is list: x = torch.stack(x)
            softmax_weight = nn.functional.softmax(self.weight, dim=-1)
            B, T, D = x.shape[1], x.shape[2], x.shape[3]
            x = x.reshape(self.num_layers, -1)
            x = torch.matmul(softmax_weight, x).reshape(B, T, D)
        # Select a specific layer
        elif self.select_layer != -1:
            x = x[self.select_layer]

        if self.spec_aug and not self.spec_aug_prev and self.model.training:
            x = spec_augment(x, mask_T=70, mask_F=86, num_T=2, num_F=2, p=1.0) # (B, T, D)

        # If using a downsampling model, apply tile and padding
        if self.dr > 1:
            x = self.upsample(x, input_len) # (B, T, D)

        # permute to output
        if self.permute_input:
            x = x.permute(1, 0, 2).contiguous() # (B, T, D) -> (T, B, D)

        return x # (B, T, D) or (T, B, D)


    def forward(self, x):
        if self.no_grad:
            with torch.no_grad():
                self.model.eval()
                x = self._forward(x)
        else:
            x = self._forward(x)
        return x


#######################
# POSITIONAL ENCODING #
#######################
MAX_SEQLEN = 9000
@lru_cache(maxsize=1)
def get_sinusoid_table(hidden_size):
    def _cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)
    def _get_posi_angle_vec(position):
        return [_cal_angle(position, hid_j) for hid_j in range(hidden_size)]
    sinusoid_table = np.array([_get_posi_angle_vec(pos_i) for pos_i in range(MAX_SEQLEN)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def position_encoding(seq_len, hidden_size):
    """ position encoding table """
    table = get_sinusoid_table(hidden_size)[:seq_len]
    # no extra CPU and GPU memory allocation
    # after getting the (seq_len, hidden_size) tensor, one should first put
    # this tensor into GPU then expand it
    return table  # (seq_len, hidden_size)


################
# SPEC AUGMENT #
################
"""
Process training data for the supervised ASR model by
masking to time-steps and channels during training
which delays overfitting and significantly improves the final accuracy numbers.
Input:
    `spec`: input real frames, with shape: (batch_size, seq_len, feature_dim)
    `mask_T`: the time mask parameter T described in the SpecAugment paper,
              we use default values based on the LD Policy
              (In paper: T=100, we use 70 since we are training on the 100 hr subset only)
    `mask_F`: the frequency mask parameter F described in the SpecAugment paper,
              we use default values based on the LD Policy
              (In paper: F=27:D=80*3 -> F=4.5:D=40, where D is acoustic dimension)
    `num_T` : the number of time masks applied (In paper: mT=2)
    `num_F` : the number of frequency masks applied (In paper: mF=2)
    `p` : upper bound ratio (In paper: p=1.0)
Output:
    `spec`: augmented frames, with shape: (batch_size, seq_len, feature_dim)
"""
def spec_augment(spec, mask_T=70, mask_F=4, num_T=2, num_F=2, p=1.0):

    def _start_to_intervals(starts, consecutive):
        tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
        offset = torch.arange(consecutive).expand_as(tiled)
        intervals = tiled + offset
        return intervals.view(-1)

    with torch.no_grad():
        upper_bound = spec.shape[1] * p # upper bound on the time mask so that a time mask cannot be wider than p times the number of time steps

        for idx in range(spec.shape[0]):

            # time masking
            if mask_T > 0 and mask_T < upper_bound:
                for _ in range(num_T):
                    rand_consecutive = random.randint(0, mask_T)
                    chosen_start = torch.randperm(spec.shape[1] - rand_consecutive)[:1]
                    chosen_intervals = _start_to_intervals(chosen_start, rand_consecutive)
                    spec[idx, chosen_intervals, :] = 0

            # frequency masking
            if mask_F > 0:
                for _ in range(num_F):
                    rand_bandwidth = random.randint(0, mask_F)
                    chosen_start = torch.randperm(spec.shape[2] - rand_bandwidth)[:1]
                    chosen_intervals = _start_to_intervals(chosen_start, rand_bandwidth)
                    spec[idx, :, chosen_intervals] = 0

        return spec


#######
# LIN #
#######
"""
Linear Input Networks (LIN) for domain adaptation
Params:
    `options`: a python dictionary containing arguments for pytorch kaldi, give None if not using with pytorch-kaldi:
    `intput_dim`: int, input dimension of model
"""
class LIN(nn.Module):
    def __init__(self, options, inp_dim):
        super(LIN, self).__init__()

        self.out_dim = inp_dim # This attribute is for pytorch-kaldi
        self.linear = nn.Linear(inp_dim, inp_dim)
        self.linear.weight.data.copy_(torch.eye(inp_dim))

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.linear = self.linear.to(self.device)
        self.linear.train()

    def forward(self, x):
        x = self.linear(x)
        return x
