# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .RevIN import RevIN

import math
from torch.fft import rfft, irfft


class backbone_new_Linear(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, head_dropout=0, individual=False, revin=True,
                 affine=True, subtract_last=False,
                 verbose: bool = False, **kwargs):
        super().__init__()
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.head_nf = context_window * (context_window // 2 + 1)
        self.n_vars = c_in
        self.individual = individual

        self.head = Flatten_Head_Linear(self.individual, self.n_vars, self.head_nf, target_window,
                                        head_dropout=head_dropout)

        sr = context_window
        ts = 1.0 / sr
        t = np.arange(0, 1, ts)
        t = torch.tensor(t).cuda()
        for i in range(context_window // 2 + 1):
            if i == 0:
                cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
                sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
            else:
                cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
                sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])

        self.cos = nn.Parameter(cos, requires_grad=False)
        self.sin = nn.Parameter(sin, requires_grad=False)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        norm = z.size()[-1]
        # (bs * nvars * seq_len) -> (bs * nvars * components)
        frequency = rfft(z, axis=-1)
        X_oneside = frequency / (norm) * 2

        basis_cos = torch.einsum('bkp,pt->bkpt', X_oneside.real, self.cos)
        basis_sin = torch.einsum('bkp,pt->bkpt', X_oneside.imag, self.sin)

        z = basis_cos + basis_sin
        z = self.head(z)

        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)

        return z


class Flatten_Head_Linear(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
        return x


class backbone_new_NLinear(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, head_dropout=0, individual=False, revin=True,
                 affine=True, subtract_last=False,
                 verbose: bool = False, **kwargs):
        super().__init__()
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.head_nf = context_window * (context_window // 2 + 1)
        self.n_vars = c_in
        self.individual = individual

        self.head = Flatten_Head_NLinear(self.individual, self.n_vars, self.head_nf, target_window,
                                         head_dropout=head_dropout)

        sr = context_window
        ts = 1.0 / sr
        t = np.arange(0, 1, ts)
        t = torch.tensor(t).cuda()
        for i in range(context_window // 2 + 1):
            if i == 0:
                cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
                sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
            else:
                cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
                sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])

        self.cos = nn.Parameter(cos, requires_grad=False)
        self.sin = nn.Parameter(sin, requires_grad=False)

    def forward(self, z, return_repr):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        norm = z.size()[-1]
        frequency = rfft(z, axis=-1)
        X_oneside = frequency / (norm) * 2

        basis_cos = torch.einsum('bkp,pt->bkpt', X_oneside.real, self.cos)
        basis_sin = torch.einsum('bkp,pt->bkpt', X_oneside.imag, self.sin)

        z = basis_cos + basis_sin
        z = z.float()
        z = self.head(z)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        if not return_repr:
            return z
        else:
            return z,


class Flatten_Head_NLinear(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Sequential(
                nn.Linear(nf, 720 * 2),
                nn.Dropout(p=0.15),
                nn.ReLU(),
                nn.Linear(720 * 2, 720 * 2),
                nn.Dropout(p=0.15),
                nn.ReLU(),
                nn.Linear(720 * 2, target_window)
            )

    def forward(self, x, return_repr: bool):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
        if not return_repr:
            return x
        else:
            return x

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            # self.linear = nn.Sequential(
            #                                 nn.Linear(nf,720*2),
            #                                 nn.Dropout(p=0.15),
            #                                 nn.ReLU(),
            #                                 nn.Linear(720*2, 720*2),
            #                                 nn.Dropout(p=0.15),
            #                                 nn.ReLU(),
            #                                 nn.Linear(720*2, target_window)
            #                             )

            self.linear = nn.Linear(nf, target_window)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
        return x
