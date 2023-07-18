import copy
import math

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
# multi-head attention
from torch.nn import MultiheadAttention

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
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, y, x_mask=None, y_mask=None):
        """
        x: (batch_size, seq_len_1, d_model)
        y: (batch_size, seq_len_2, d_model)
        """
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        x = self.multihead_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=y_mask,
        )[0]
        x = self.dropout(x)
        x = self.norm(x + self.linear(x))
        return x.transpose(0, 1)

# from https://pytorch.org/docs/1.13/_modules/torch/nn/modules/transformer.html#TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        return_additional_layers=None,
        cross_attention=False,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        if cross_attention:
            attn = CrossAttention(
                encoder_layer.d_model,
                encoder_layer.n_heads,
                dropout=encoder_layer.dropout,
            )
            self.attentions = nn.ModuleList([copy.deepcopy(attn) for _ in range(num_layers)])
        self.cross_attention = cross_attention
        self.num_layers = num_layers
        self.norm = norm
        self.return_additional_layers = return_additional_layers

    def forward(self, src, mask=None, src_key_padding_mask=None, t_condition=None, c_condition=None, need_weights=False):
            if src_key_padding_mask is not None:
                _skpm_dtype = src_key_padding_mask.dtype
                if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                    raise AssertionError("only bool and floating types of key_padding_mask are supported")
            
            output = src
            src_key_padding_mask_for_layers = src_key_padding_mask

            output_for_return = []

            if need_weights:
                weight_list = []

            for i, mod in enumerate(self.layers):
                if t_condition is not None:
                    output = output + t_condition
                if need_weights:
                    output, weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers, need_weights=need_weights)
                    weight_list.append(weights)
                else:
                    output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)
                if self.return_additional_layers is not None and i in self.return_additional_layers:
                    output_for_return.append(output)

            if self.norm is not None:
                output = self.norm(output)

            if self.return_additional_layers is not None:
                if need_weights:
                    return {
                        "output": output,
                        "activations": output_for_return,
                        "attention": weight_list,
                    }
                else:
                    return {
                        "output": output,
                        "activations": output_for_return,
                    }
            else:
                if need_weights:
                    return {
                        "output": output,
                        "attention": weight_list,
                    }
                else:
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

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=False):
        x = src
        if self.norm_first:
            if not need_weights:
                attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            else:
                attn, weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, need_weights=need_weights)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            if not need_weights:
                attn = self._sa_block(x, src_mask, src_key_padding_mask)
            else:
                attn, weights = self._sa_block(x, src_mask, src_key_padding_mask, need_weights=need_weights)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))
        if need_weights:
            return x, weights
        else:
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
            need_weights=False,
        ):
        if not need_weights:
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights)[0]
        else:
            x, weights = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights)
        if need_weights:
            return self.dropout1(x), weights
        else:
            return self.dropout1(x)
        
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
    ):
        super().__init__()
        
        self.in_layer = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

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
        self.hparams = {
            "in_channels": in_channels,
            "filter_size": filter_size,
            "n_heads": n_heads,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "depthwise": depthwise,
            "n_layers": n_layers,
        }

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, speaker_prompt=None, phones=None, vocex=None):
        padding_mask = x.sum(dim=-1) != 0
        padding_mask = padding_mask.to(x.dtype)

        out = self.in_layer(x)
        out = self.positional_encoding(out)
        res = self.layers(out, src_key_padding_mask=mel_padding_mask)
        out = self.linear(out_conv)
        out = out.transpose(1, 2)

        measure_results = {}
        measure_true = {}
        loss_dict = {}
        for i, measure in enumerate(self.measures):
            measure_out = out[:, i]
            if not measure.endswith("_binary") and not self.scalers[measure].is_fit:
                self.scalers[measure].partial_fit(measures[measure])
            measure_results[measure] = measure_out
        if measures is not None:
            loss_dict = {}
            for measure in self.measures:
                if not measure.endswith("_binary"):
                    measure_true[measure] = self.scalers[measure].transform(measures[measure])
                else:
                    measure_true[measure] = measures[measure]
            measures_losses = []
            for measure in self.measures:
                if measure.endswith("_binary"):
                    m_loss = nn.BCEWithLogitsLoss()(measure_results[measure]*mel_padding_mask, measure_true[measure]*mel_padding_mask)
                else:
                    if not self.use_softdtw:
                        m_loss = nn.MSELoss()(measure_results[measure]*mel_padding_mask, measure_true[measure]*mel_padding_mask)
                    else:
                        if self.verbose:
                            print(measure_results[measure], measure_true[measure])
                        m_loss = self.softdtw(
                            measure_results[measure]*mel_padding_mask,
                            measure_true[measure]*mel_padding_mask,
                        ) / 1000
                loss_dict[measure] = m_loss
                measures_losses.append(m_loss)
            loss = sum(measures_losses) / len(self.measures)
        else:
            loss = None
        ### d-vector
        # predict d-vector using global average and max pooling as input
        out_conv_dvec = self.dvector_conv_in_layer(out_conv)
        x = self.dvector_x_in_layer(x)
        out_dvec = self.dvector_layers(out_conv_dvec + x, src_key_padding_mask=mel_padding_mask)
        dvector_input = torch.cat(
            [
                torch.mean(out_dvec, dim=1),
                torch.max(out_dvec, dim=1)[0],
            ],
            dim=1,
        )
        dvector_pred = self.dvector_linear(dvector_input)
        if dvector is not None:
            if not self.scalers["dvector"].is_fit:
                self.scalers["dvector"].partial_fit(dvector)
            true_dvector = self.scalers["dvector"].transform(dvector)
            dvector_loss = nn.L1Loss()(dvector_pred, true_dvector)
            loss_dict["dvector"] = dvector_loss
            if loss is not None:
                loss += dvector_loss
                loss /= 2
            else:
                loss = dvector_loss
        if (not inference) and not (hasattr(self, "onnx_export") and self.onnx_export):
            results = {
                "loss": loss,
                "compound_losses": loss_dict,
            }
            if return_activations:
                results["activations"] = [a.detach() for a in activations]
            if return_attention:
                results["attention"] = [a.detach() for a in attention]
            return results
        else:
            # transform back to original scale
            for measure in self.measures:
                if not measure.endswith("_binary"):
                    measure_results[measure] = self.scalers[measure].inverse_transform(
                        measure_results[measure]
                    )
                else:
                    measure_results[measure] = torch.sigmoid(measure_results[measure]).detach()
            dvector_pred = self.scalers["dvector"].inverse_transform(dvector_pred)
            if is_onnx:
                return [measure_results[measure] for measure in self.measures] + [dvector_pred]
            results = {
                "loss": loss,
                "compound_losses": loss_dict,
                "measures": measure_results,
                "dvector": dvector_pred,
            }
            if return_activations:
                results["activations"] = [a.detach() for a in activations]
            if return_attention:
                results["attention"] = [a.detach() for a in attention]
            return results