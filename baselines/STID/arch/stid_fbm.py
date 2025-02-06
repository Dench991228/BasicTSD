import math
import numpy as np
import torch
from torch import nn
from torch.fft import rfft

from baselines.FBM.arch.FBP_backbone import Flatten_Head_NLinear
from baselines.FBM.arch.RevIN import RevIN
from baselines.STID.arch import STID


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

    def forward(self, z):  # z: [bs x nvars x seq_len]
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

        return z

class STID_FBM(STID):
    def __init__(self, **model_args):
        # 先创建父类的样本
        super().__init__(**model_args)
        # 再创建FBM部分
        revin = True
        affine = True
        subtract_last = False
        individual = False
        context_window = model_args['input_len']
        c_in = model_args['num_nodes']
        target_window = model_args['output_len']
        self.model = backbone_new_NLinear(c_in=c_in,
                                          context_window = context_window,
                                          target_window=target_window,
                                          head_dropout=0,
                                          individual=individual,
                                          revin=revin,
                                          affine=affine,
                                          subtract_last=subtract_last,
                                          verbose=False)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        parent_prediction = super().forward(history_data, future_data, batch_seen, epoch, train, **kwargs)
        x = history_data[:, :, :, 0]
        x = x.permute(0, 2, 1)
        fbm_prediction = self.model(x)
        fbm_prediction = fbm_prediction.permute(0, 2, 1).unsqueeze(-1)
        return fbm_prediction + parent_prediction