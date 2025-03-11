from typing import Dict

import torch.nn as nn
import torch

from baselines.MyModel.arch.DecomposeFormer import Embedder, DecomposeFormer_layer
from baselines.MyModel.arch.MyFormer import moving_avg



class DecomposeFormer_adv(nn.Module):
    """
    这一个模型的设计原理与之前的DecomposeFormer基本一致，区别在于会增加一个额外的Trend损失函数，相当于是规约我的趋势部分的预测结果
    """
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        **kwargs
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        # note 嵌入的部分
        self.trend_embedding = Embedder(num_nodes, in_steps, input_dim, input_embedding_dim, steps_per_day, tod_embedding_dim,
                                        dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim)
        self.seasonal_embedding = Embedder(num_nodes, in_steps, input_dim, input_embedding_dim, steps_per_day, tod_embedding_dim,
                                        dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim)

        # note 回归头
        if use_mixed_proj:
            self.output_proj_trend = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
            self.output_proj_seasonal = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim*2, self.output_dim)
        self.decomposer = moving_avg(kernel_size=3, stride=1)
        self.attn_layers = nn.ModuleList(
            [
                DecomposeFormer_layer(self.model_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int,
                epoch: int, train: bool, return_repr: bool = False, **kwargs) -> torch.Tensor|Dict:
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
        # note 先进行序列分解
        trend_components = self.decomposer(history_data[:, :, :, 0])
        assert trend_components.shape == history_data[:, :, :, 0].shape
        seasonal_components = history_data[:, :, :, 0] - trend_components
        # (bs, T, num_nodes, input_dim)
        seasonal_components = torch.cat([seasonal_components.unsqueeze(-1), history_data[:, :, :, 1:]], dim=-1)
        trend_components = torch.cat([trend_components.unsqueeze(-1), history_data[:, :, :, 1:]], dim=-1)
        #print("after decompose",trend_components.shape, seasonal_components.shape)
        # (bs, T, num_nodes, model_dim)
        trend_embeddings = self.trend_embedding(trend_components)
        seasonal_embeddings = self.seasonal_embedding(seasonal_components)
        #print("after embedding", trend_components.shape, seasonal_components.shape)
        batch_size = x.shape[0]

        x = torch.stack([trend_embeddings, seasonal_embeddings], dim=-2)
        #print("before deep", x.shape)
        # note 花式变换 (batch_size, in_steps, num_nodes, 2, model_dim)
        for attn in self.attn_layers:
            x = attn(x)
        # (batch_size, in_steps, num_nodes, model_dim)
        # note 整理一下需要输出的表示
        # (batch_size, num_nodes, in_steps, model_dim)
        repr = x.reshape(batch_size, self.in_steps, self.num_nodes, self.model_dim * 2).transpose(1, 2)
        repr = repr.reshape(batch_size, -1, self.model_dim * self.in_steps * 2)
        #print(repr.shape)
        out = x.transpose(1, 2)  # (batch_size, in_steps, num_nodes, 2, model_dim)
        # note 得到趋势部分的预测
        out_trend = out[:, :, :, 0, :].reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        pred_trend = self.output_proj_trend(out_trend).view(
            batch_size, self.num_nodes, self.in_steps, self.output_dim
        )
        pred_trend = pred_trend.transpose(1, 2)
        # note 得到季节部分的预测
        out_seasonal = out[:, :, :, 1, :].reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        pred_seasonal = self.output_proj_seasonal(out_seasonal).view(
            batch_size, self.num_nodes, self.in_steps, self.output_dim
        )
        pred_seasonal = pred_seasonal.transpose(1, 2)
        out = pred_trend + pred_seasonal
        future_trend = self.decomposer(future_data[:, :, :, 0]).unsqueeze(-1)
        if not return_repr:
            return {
                "prediction": out,
                "prediction_trend": pred_trend,
                "target_trend": future_trend,
                "input_trend": trend_components[:, :, :, :1],
            }
        else:
            return {
                "prediction": out,
                "representation": repr,
                "prediction_trend": pred_trend,
                "target_trend": future_trend,
                "input_trend": trend_components[:, :, :, :1],
            }
