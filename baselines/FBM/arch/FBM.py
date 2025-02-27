import torch
import torch.nn as nn

from baselines.FBM.arch.FBP_backbone import backbone_new_NLinear

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        front = x[:, 0:1, :].repeat(1,1, 1)
        end = x[:, -1:, :].repeat(1, 1, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class FBM(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in,
                 decomposition: bool = False,
                 kernel_size: int = 3,
                 verbose: bool = False,
                 revin: bool = True):
        super().__init__()
        # load parameters
        c_in = enc_in
        context_window = seq_len
        target_window = pred_len
        revin = revin
        affine = True
        subtract_last = False
        individual = False
        head_dropout = 0
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.linear=nn.Linear(context_window, target_window)
            self.model_trend = backbone_new_NLinear(c_in=c_in, context_window = context_window, target_window=target_window,head_dropout=head_dropout, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose)
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
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        x = x.unsqueeze(-1)
        return x

