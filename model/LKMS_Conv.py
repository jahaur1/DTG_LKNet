__all__ = ['LKMS_Conv']

import copy
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from Patch_layers import *

class LKMS_Conv(nn.Module):
    def __init__(self, c_in: int, seq_len: int, context_window: int, patch_len: int, stride: int,
                 n_layers: int = 6, dw_ks=[9,11,15,21,29,39], d_model=64, d_ff: int = 256, norm: str = 'batch',
                 dropout: float = 0., act: str = "gelu",
                 padding_patch=None, deformable=False, enable_res_param=True, re_param=True, re_param_kernel=3):

        super().__init__()

        self.deformable = deformable
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        seq_len = (patch_num - 1) * self.stride + self.patch_len
        if deformable:
            self.deformable_sampling = DepatchSampling(c_in, seq_len, self.patch_len, self.stride)


        if act.lower() == "relu":
            self.activation = nn.ReLU()
        elif act.lower() == "gelu":
            self.activation = nn.GELU()
        elif callable(act):
            self.activation = act()
        else:
            raise ValueError(f'{act} is not available. You can use "relu", "gelu", or a callable')


        self.backbone = ConvBackbone(
            patch_num=patch_num,
            patch_len=patch_len,
            kernel_size=dw_ks,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            norm=norm,
            dropout=dropout,
            activation=self.activation,
            enable_res_param=enable_res_param,
            re_param=re_param,
            re_param_kernel=re_param_kernel,
            device='cuda:0'
        )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        if not self.deformable:
            z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            z = self.deformable_sampling(z)

        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]
        z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = z.permute(0, 1, 3, 2)
        return z


class ConvBackbone(nn.Module):
    def __init__(self, patch_num, patch_len, kernel_size=[11, 15, 21, 29, 39, 51], n_layers=6, d_model=128,
                 d_ff=256, norm='batch', dropout=0., activation=None, enable_res_param=True,
                 re_param=True, re_param_kernel=3, device='cuda:0'):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input embedding
        self.W_P = nn.Linear(patch_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            ConvLayer(
                d_model,
                d_ff=d_ff,
                kernel_size=kernel_size[i],
                dropout=dropout,
                activation=activation,
                enable_res_param=enable_res_param,
                norm=norm,
                re_param=re_param,
                small_ks=re_param_kernel,
                device=device
            ) for i in range(n_layers)
        ])

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]

        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]

        output = u.permute(0, 2, 1)  # [bs * nvars x d_model x patch_num]
        for mod in self.layers:
            output = mod(output)
        z = output.permute(0, 2, 1)  # z: [bs * nvars x patch_num x d_model]

        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z


class SublayerConnection(nn.Module):
    def __init__(self, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            return x + self.dropout(self.a * out_x)


class ConvLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 256, kernel_size: int = 9, dropout: float = 0.1,
                 activation=None, enable_res_param=True, norm='batch', re_param=True, small_ks=3, device='cuda:0'):
        super(ConvLayer, self).__init__()

        self.norm_tp = norm
        self.re_param = re_param

        if not re_param:
            self.DW_conv = nn.Conv1d(d_model, d_model, kernel_size, 1, 'same', groups=d_model)
        else:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)
            self.DW_conv_small = nn.Conv1d(d_model, d_model, small_ks, stride=1, padding='same', groups=d_model)
            self.DW_infer = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)

        self.dw_act = activation

        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)


        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1, 1),
            activation,
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, 1, 1)
        )

        # Add & Norm
        self.sublayerconnect2 = SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

    def _get_merged_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        module_output = copy.deepcopy(self.DW_conv_large)
        module_output.weight = torch.nn.Parameter(
            module_output.weight + F.pad(self.DW_conv_small.weight, (left_pad, right_pad), value=0))
        module_output.bias = torch.nn.Parameter(module_output.bias + self.DW_conv_small.bias)
        self.DW_infer = module_output

    def forward(self, src: torch.Tensor) -> torch.Tensor:  # [B, C, L]
        ## Deep-wise Conv Layer
        if not self.re_param:
            out_x = self.DW_conv(src)
        else:
            if self.training:  # training phase
                large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                out_x = large_out + small_out
            else:  # testing phase
                self._get_merged_param()
                out_x = self.DW_infer(src)

        src2 = self.dw_act(out_x)
        src = self.sublayerconnect1(src, src2)

        # Norm
        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
        src = self.dw_norm(src)
        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src

        ## Position-wise Conv Feed-Forward
        src2 = self.ff(src)
        src2 = self.sublayerconnect2(src, src2)  # Add: residual connection with residual dropout

        # Norm
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2
        src2 = self.norm_ffn(src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2

        return src2
