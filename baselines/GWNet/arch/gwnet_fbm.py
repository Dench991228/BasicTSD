import torch

from baselines.FBM.arch.FBP_backbone import backbone_new_NLinear
from baselines.GWNet.arch import GraphWaveNet


class GraphWavenet_FBM(GraphWaveNet):
    def __init__(self, num_nodes, dropout=0.3, supports=None,
                    gcn_bool=True, addaptadj=True, aptinit=None,
                    in_dim=2, out_dim=12, residual_channels=32,
                    dilation_channels=32, skip_channels=256, end_channels=512,
                    kernel_size=2, blocks=4, layers=2):
        super().__init__(num_nodes, dropout, supports, gcn_bool, addaptadj, aptinit,
                         in_dim, out_dim, residual_channels, dilation_channels, skip_channels,
                         end_channels, kernel_size, blocks, layers)
        revin = True
        affine = True
        subtract_last = False
        individual = False
        context_window = out_dim
        c_in = num_nodes
        target_window = out_dim
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