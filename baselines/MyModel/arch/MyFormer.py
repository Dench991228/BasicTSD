from typing import Dict

import torch
import torch.nn as nn

from baselines.STAEformer.arch import STAEformer

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # (Bs, T, Num_nodes)
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # (Bs, T, num_nodes) -> (bs, num_nodes, T) 平均池化
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class MyFormer(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, steps_per_day=288, input_dim=3, output_dim=1,
                 input_embedding_dim=24, tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
                 adaptive_embedding_dim=80, feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1,
                 use_mixed_proj=True, decompose_type: str="MA", *args, **kwargs):

        # note 这个方法旨在对原始输入进行分解，然后再进行transformer变换，得到最终的结果
        super().__init__(*args, **kwargs)
        self.seasonal_branch = STAEformer(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            input_dim=input_dim,
            output_dim=output_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_mixed_proj=use_mixed_proj,
        )
        self.trend_branch = STAEformer(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            input_dim=input_dim,
            output_dim=output_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_mixed_proj=use_mixed_proj,
        )
        self.output_dim = output_dim
        self.out_steps = out_steps
        self.regression_head_trend = nn.Linear(self.trend_branch.model_dim * in_steps, out_steps * output_dim)
        self.regression_head_seasonal = nn.Linear(self.trend_branch.model_dim * in_steps, out_steps * output_dim)
        self.decomposer = moving_avg(kernel_size=3, stride=1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int,
                epoch: int, train: bool, return_repr: bool = False, **kwargs) -> torch.Tensor|Dict:
        """
        :return 形如 (batch_size, out_steps, num_nodes, output_dim) 的张量
        """
        # note 先进行序列分解
        trend_components = self.decomposer(history_data[:, :, :, 0])
        assert trend_components.shape == history_data[:, :, :, 0].shape
        seasonal_components = history_data[:, :, :, 0] - trend_components
        seasonal_components = torch.cat([seasonal_components.unsqueeze(-1), history_data[:, :, :, 1:]], dim=-1)
        trend_components = torch.cat([trend_components.unsqueeze(-1), history_data[:, :, :, 1:]], dim=-1)
        # note 再进行分别建模
        # (bs, num_node, in_steps * model_dim) * 2
        trend_repr = self.trend_branch(history_data=trend_components,
                                       future_data=future_data,
                                       batch_seen=batch_seen,
                                       epoch=epoch,
                                       return_repr=True,
                                       train=train,
                                       **kwargs)['representation']
        seasonal_repr = self.seasonal_branch(history_data=seasonal_components,
                                             future_data=future_data,
                                             batch_seen=batch_seen,
                                             epoch=epoch,
                                             return_repr=True,
                                             train=train,
                                             **kwargs)['representation']
        bs, num_nodes, _ = trend_repr.shape
        # (bs, num_node, in_steps * model_dim * 2) -> (bs, num_node, out_steps * output_dim)
        final_repr = torch.cat([trend_repr, seasonal_repr], dim=-1)
        output_trend = self.regression_head_trend(trend_repr).transpose(-1, -2).unsqueeze(-1)
        output_seasonal = self.regression_head_seasonal(seasonal_repr).transpose(-1, -2).unsqueeze(-1)
        output = output_trend + output_seasonal
        print(output.shape)
        future_trend = self.decomposer(future_data[:, :, :, 0])
        if return_repr:
            return {
                "prediction": output,
                "representation": final_repr,
                "input_trend": trend_components[:, :, :, :1],
                "target_trend": future_trend.unsqueeze(-1),
                "prediction_trend": output_trend,
            }
        else:
            return {
                "prediction": output,
                "input_trend": trend_components[:, :, :, :1],
                "target_trend": future_trend.unsqueeze(-1),
                "prediction_trend": output_trend,
            }