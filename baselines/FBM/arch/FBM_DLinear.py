import torch
import torch.nn as nn

from baselines.FBM.arch.FBM import series_decomp
from baselines.FBM.arch.FBP_backbone import backbone_new_NLinear


class FBM_DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, decomposition: bool = False, kernel_size: int = 3, verbose: bool = False):
        super().__init__()
        # load parameters
        c_in = enc_in
        context_window = seq_len
        target_window = pred_len
        revin = True
        affine = True
        subtract_last = False
        individual = False
        head_dropout = 0
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.linear=nn.Linear(context_window, target_window)
            self.model_res = backbone_new_NLinear(c_in=c_in, context_window = context_window, target_window=target_window,head_dropout=head_dropout, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose)
        else:
            self.model = backbone_new_NLinear(c_in=c_in, context_window = context_window, target_window=target_window,head_dropout=head_dropout, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):  # x: [Batch, Input length, Channel]
        # history_data: [B, T, num_variate, num_features] -> [B, T, num_variate]
        x = history_data[:, :, :, 0]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.linear(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        x = x.unsqueeze(-1)
        return x

