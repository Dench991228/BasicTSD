import torch
import torch.nn as nn

from baselines.FBM.arch.FBP_backbone import backbone_new_NLinear


class FBM(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, verbose: bool = False):
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
        self.model = backbone_new_NLinear(c_in=c_in, context_window=context_window, target_window=target_window,
                            head_dropout=head_dropout, individual=individual, revin=revin,
                            affine=affine,
                            subtract_last=subtract_last, verbose=verbose)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):  # x: [Batch, Input length, Channel]
        # history_data: [B, T, num_variate, num_features] -> [B, T, num_variate]
        x = history_data[:, :, :, 0]
        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        x = x.unsqueeze(-1)
        return x

