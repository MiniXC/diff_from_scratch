import copy
import math

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
# multi-head attention
from torch.nn import MultiheadAttention
import numpy as np
import argparse

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)
        return self.dropout(x)

class CrossAttention(nn.Module):
    """
    Cross Attention between two sequences
    x: (batch_size, seq_len_1, d_model)
    y: (batch_size, seq_len_2, d_model)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, y, mask):
        """
        x: (batch_size, seq_len_1, d_model)
        y: (batch_size, seq_len_2, d_model)
        """
        # attn_mask (batch_size, seq_len_1, seq_len_2)
        # construct attn_mask
        x, _ = self.multihead_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=mask,
            need_weights=False,
        )
        x = self.dropout(x)
        x = self.norm(x + self.linear(x))
        return x

# from https://pytorch.org/docs/1.13/_modules/torch/nn/modules/transformer.html#TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        return_additional_layers=None,
        d_model=None,
        n_heads=None,
        dropout=None,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        attn = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.attentions = nn.ModuleList([copy.deepcopy(attn) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_additional_layers = return_additional_layers

    def forward(self, src, mask=None, src_key_padding_mask=None, t_condition=None, c_condition=None, c_mask=None):            
            output = src

            for i, mod in enumerate(self.layers):
                output = output + t_condition
                output = self.attentions[i](output, c_condition, c_mask)
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            output = self.norm(output)

            return output

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class ConformerLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        old_kwargs = {k: v for k, v in kwargs.items() if "conv_" not in k}
        super().__init__(*args, **old_kwargs)
        del self.linear1
        del self.linear2
        if "conv_depthwise" in kwargs and kwargs["conv_depthwise"]:
            self.conv1 = DepthwiseConv1d(
                kwargs["conv_in"],
                kwargs["conv_filter_size"],
                kernel_size=kwargs["conv_kernel"][0],
                padding=(kwargs["conv_kernel"][0] - 1) // 2,
            )
            self.conv2 = DepthwiseConv1d(
                kwargs["conv_filter_size"],
                kwargs["conv_in"],
                kernel_size=kwargs["conv_kernel"][1],
                padding=(kwargs["conv_kernel"][1] - 1) // 2,
            )
        else:
            self.conv1 = nn.Conv1d(
                kwargs["conv_in"],
                kwargs["conv_filter_size"],
                kernel_size=kwargs["conv_kernel"][0],
                padding=(kwargs["conv_kernel"][0] - 1) // 2,
            )
            self.conv2 = nn.Conv1d(
                kwargs["conv_filter_size"],
                kwargs["conv_in"],
                kernel_size=kwargs["conv_kernel"][1],
                padding=(kwargs["conv_kernel"][1] - 1) // 2,
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            attn = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        x = self.conv2(
            self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        ).transpose(1, 2)
        return self.dropout2(x)

    def _sa_block(
            self, 
            x,
            attn_mask,
            key_padding_mask=None,
        ):
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
            return self.dropout1(x)

class TimestepEmbedding(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.fc_t1 = nn.Linear(
            dim_in,
            dim_hidden,
        )
        self.fc_t2 = nn.Linear(
            dim_hidden,
            dim_out,
        )
        self.dim_in = dim_in

        # init
        nn.init.xavier_uniform_(self.fc_t1.weight)
        nn.init.xavier_uniform_(self.fc_t2.weight)
        nn.init.zeros_(self.fc_t1.bias)
        nn.init.zeros_(self.fc_t2.bias)

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, steps):
        half_dim = self.dim_in // 2
        _embed = np.log(10000) / (half_dim - 1)
        _embed = torch.exp(torch.arange(half_dim) * -_embed)
        _embed = steps * _embed
        diff_embed = torch.cat(
            (torch.sin(_embed), torch.cos(_embed)),
            2
        ).to(steps.device)
        diff_embed = TimestepEmbedding.swish(self.fc_t1(diff_embed))
        diff_embed = TimestepEmbedding.swish(self.fc_t2(diff_embed))
        return diff_embed
        
class ConformerModel(nn.Module):

    def __init__(
        self,
        in_channels=80,
        filter_size=512,
        n_heads=4,
        kernel_size=3,
        dropout=0.1,
        depthwise=True,
        n_layers=8,
        sample_size=(512, 80),
    ):
        super().__init__()
        
        self.in_layer = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.time_embedding = TimestepEmbedding(10, filter_size, filter_size)

        self.layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                n_heads,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=depthwise,
            ),
            num_layers=n_layers,
            d_model=filter_size,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.linear = nn.Sequential(
            nn.Linear(filter_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, in_channels),
        )
        
        self.apply(self._init_weights)

        # save hparams
        self.config = {
            "in_channels": in_channels,
            "filter_size": filter_size,
            "n_heads": n_heads,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "depthwise": depthwise,
            "n_layers": n_layers,
            "sample_size": sample_size,
        }
        self.config = argparse.Namespace(**self.config)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, timestep, t_condition=None, c_condition=None, c_mask=None):
        padding_mask = x.sum(dim=-1) != 0
        padding_mask = padding_mask.to(x.dtype)

        step_embed = self.time_embedding(timestep)

        t_condition = t_condition + step_embed

        out = self.in_layer(x)
        out = self.positional_encoding(out)

        out = self.layers(
            out,
            src_key_padding_mask=padding_mask,
            t_condition=t_condition,
            c_condition=c_condition,
            c_mask=c_mask
        )
        out = self.linear(out)
        out = out

        return out