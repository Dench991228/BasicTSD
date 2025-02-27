import torch

from baselines.FBM.arch.FBP_backbone import backbone_new_NLinear
from baselines.STAEformer.arch import STAEformer


class STAEformer_FBM(STAEformer):
    def __init__(self, **model_args):
        # 先创建父类
        super().__init__(**model_args)
        # 再创建剩下的
        revin = True
        affine = True
        subtract_last = False
        individual = False
        context_window = model_args['in_steps']
        c_in = model_args['num_nodes']
        target_window = model_args['out_steps']
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
