""" TDNNF architecture"""
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>

import torch
import torch.nn as nn
import torch.nn.functional as F
from pkwrap.nn import NaturalAffineTransform, TDNNF


class TDNNFBatchNorm(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        bottleneck_dim=160,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        bypass_scale=0.66,
        p_dropout=0.1
    ):
        super(TDNNFBatchNorm, self).__init__()
        self.tdnn = TDNNF(
            feat_dim,
            output_dim,
            bottleneck_dim,
            context_len=context_len,
            subsampling_factor=subsampling_factor,
            orthonormal_constraint=orthonormal_constraint,
            bypass_scale=bypass_scale
        )
        self.bn = nn.BatchNorm1d(output_dim, affine=False)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, input):
        mb, T, D = input.shape
        x = self.tdnn(input)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.drop(x)
        return x


# Create a network like the above one
class ChainModel(nn.Module):

    def __init__(
        self,
        feat_dim,
        output_dim,
        padding=27,
        ivector_dim=0,
        hidden_dim=1024,
        bottleneck_dim=128,
        prefinal_bottleneck_dim=256,
        kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        frame_subsampling_factor=3,
        p_dropout=0.1
    ):
        super().__init__()

        # at present, we support only frame_subsampling_factor to be 3
        assert frame_subsampling_factor == 3

        assert len(kernel_size_list) == len(subsampling_factor_list)
        num_layers = len(kernel_size_list)
        input_dim = feat_dim + ivector_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_subsampling = frame_subsampling_factor

        self.padding = padding
        self.frame_subsampling_factor = frame_subsampling_factor
        self.tdnn = TDNNFBatchNorm(
            input_dim, hidden_dim,
            bottleneck_dim=bottleneck_dim,
            context_len=kernel_size_list[0],
            subsampling_factor=subsampling_factor_list[0],
            orthonormal_constraint=-1.0,
            p_dropout=p_dropout
        )
        tdnnfs = []
        for i in range(1, num_layers):
            kernel_size = kernel_size_list[i]
            subsampling_factor = subsampling_factor_list[i]
            layer = TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=bottleneck_dim,
                context_len=kernel_size,
                subsampling_factor=subsampling_factor,
                orthonormal_constraint=-1.0,
                p_dropout=p_dropout
            )
            tdnnfs.append(layer)

        # tdnnfs requires [N, C, T]
        self.tdnnfs = nn.ModuleList(tdnnfs)

        # prefinal_l affine requires [N, C, T]
        self.prefinal_chain = TDNNFBatchNorm(
            hidden_dim, hidden_dim,
            bottleneck_dim=prefinal_bottleneck_dim,
            context_len=1,
            orthonormal_constraint=-1.0,
            p_dropout=0.0
        )
        self.prefinal_xent = TDNNFBatchNorm(
            hidden_dim, hidden_dim,
            bottleneck_dim=prefinal_bottleneck_dim,
            context_len=1,
            orthonormal_constraint=-1.0,
            p_dropout=0.0
        )
        self.chain_output = NaturalAffineTransform(hidden_dim, output_dim)
        self.chain_output.weight.data.zero_()
        self.chain_output.bias.data.zero_()

        self.xent_output = NaturalAffineTransform(hidden_dim, output_dim)
        self.xent_output.weight.data.zero_()
        self.xent_output.bias.data.zero_()
        self.validate_model()

    def validate_model(self):
        N = 1
        T = (10 * self.frame_subsampling_factor)
        C = self.input_dim
        x = torch.arange(N * T * C).reshape(N, T, C).float()
        nnet_output, xent_output = self.forward(x)
        assert nnet_output.shape[1] == 10

    def pad_input(self, x):
        N, T, F = x.shape
        if self.padding > 0:
            x = torch.cat(
                [
                    x[:, 0:1, :].repeat(1, self.padding, 1),
                    x,
                    x[:, -1:, :].repeat(1, self.padding, 1)
                ],
                axis=1
            )
        return x

    def forward(self, x, dropout=0.):
        # input x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
        assert x.ndim == 3
        x = self.pad_input(x)
        # at this point, x is [N, T, C]
        x = self.tdnn(x)
        # tdnnf requires input of shape [N, C, T]
        for i in range(len(self.tdnnfs)):
            x = self.tdnnfs[i](x)
        chain_prefinal_out = self.prefinal_chain(x)
        xent_prefinal_out = self.prefinal_xent(x)
        chain_out = self.chain_output(chain_prefinal_out)
        xent_out = self.xent_output(xent_prefinal_out)
        return chain_out, F.log_softmax(xent_out, dim=2)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    feat_dim = 40
    output_dim = 3456
    model = ChainModel(feat_dim=feat_dim, output_dim=output_dim, padding=27)
    N, T = 1, 150
    C = feat_dim
    x = torch.arange(N * T * C).reshape(N, T, C).float()
    nnet_output, xent_output = model(x)
    print(x.shape, nnet_output.shape, xent_output.shape)
