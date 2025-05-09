from typing import Dict

import torch
from torch import nn

from baselines.AGCRN.arch import AGCRN
from baselines.DCRNN.arch import DCRNN
from baselines.MyModel.arch.PatchFormer import PatchTSFormer_base
from baselines.SSL.arch.ssl_factrory import ssl_factory
from baselines.STAEformer.arch import STAEformer
from baselines.GWNet.arch.gwnet_arch import GraphWaveNet
from baselines.STID.arch import STID


def model_factory(**kwargs) -> nn.Module:
    model_name = kwargs['model_name']
    print(kwargs)
    if model_name == "STAEformer":
        model = STAEformer(**kwargs)
    elif model_name == "GWNet":
        model = GraphWaveNet(**kwargs)
    elif model_name == "AGCRN":
        model = AGCRN(**kwargs)
    elif model_name == "DCRNN":
        model = DCRNN(**kwargs)
    elif model_name == "STID":
        model = STID(**kwargs)
    elif model_name == "PatchTSFormer":
        model = PatchTSFormer_base(**kwargs)
    else:
        model = GraphWaveNet(**kwargs)
    return model

class SSLWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model_name = kwargs['model_name']
        ssl_name = kwargs['ssl_name']
        ssl_loss_weight = kwargs['ssl_loss_weight']
        self.ssl_metric_name = ssl_name
        self.ssl_loss_weight = ssl_loss_weight
        self.ssl_module, self.ppa_aug = ssl_factory(**kwargs)
        self.ssl_module.ssl_loss_weight = ssl_loss_weight
        self.ssl_module.ssl_metric_name = ssl_name
        self.use_future = kwargs['ssl_use_future'] if 'ssl_use_future' in kwargs else False
        self.base_model = model_factory(**kwargs)

    def _forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool, **kwargs) -> torch.Tensor|Dict:
        return self.base_model.forward(history_data, future_data, batch_seen, epoch, train, return_repr=return_repr, **kwargs)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool = False, **kwargs):
        # note 如果没有对比损失的话
        # prediction: 预测结果，repr (B, node, dim)
        original_forward_output =  self._forward(history_data, future_data, batch_seen, epoch, train, True)
        # note 考虑和对比损失有关的内容
        if self.ssl_metric_name is not None and 't_global' not in self.ssl_metric_name:
            # 先数据增强, 先交换，再增加高斯噪声
            augmented_data = self.ppa_aug(history_data)
            rand_noise = torch.randn_like(augmented_data[:, :, :, 0]).cuda()*0.01
            augmented_data[:, :, :, 0] += rand_noise
            # (B)
            tod = history_data[:, -1, 0, 1]
            # 之后再收集对比学习部分的损失
            augmented_forward_output = self._forward(augmented_data, future_data, batch_seen, epoch, train, True)
            if not self.use_future:
                ssl_loss = self.ssl_module(original_forward_output["representation"],
                                augmented_forward_output["representation"],
                                original_series=history_data[:, :, :, 0],
                                augmented_series=augmented_data[:, :, :, 0],
                                spatial_embeddings=self.get_spatial_embeddings(),
                                tod=tod
                                )
            else:
                print("use future!")
                ssl_loss = self.ssl_module(original_forward_output["representation"],
                                augmented_forward_output["representation"],
                                future_data[:, :, :, 0],
                                augmented_data[:, :, :, 0],
                                tod=tod
                            )
            original_forward_output['other_losses'] = ssl_loss
        elif 't_global' in self.ssl_metric_name:
            model_repr = original_forward_output["representation"]
            contrastive_loss = self.ssl_module(model_repr)
            original_forward_output['other_losses'] = [
                {
                    "weight": self.ssl_loss_weight,
                    "loss": contrastive_loss,
                    "loss_name": self.ssl_metric_name
                }
            ]
        if not return_repr:
            del original_forward_output['representation']
        return original_forward_output

    def get_spatial_embeddings(self):
        if hasattr(self.base_model, "get_spatial_embeddings") and callable(self.base_model.get_spatial_embeddings):
            return self.base_model.get_spatial_embeddings()
        else:
            return None
