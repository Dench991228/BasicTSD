from typing import Dict

import torch
import torch.nn as nn

from baselines.MyModel.arch.DecomposeFormer import Embedder, SelfAttentionLayer
from baselines.MyModel.arch.MyFormer import moving_avg


class DecomposeFormer_TTS_layer(nn.Module):
    def __init__(self, model_dim: int, feed_forward_dim: int, num_heads=8, dropout=0., mask=False):
        super().__init__()
        # note 空间变换的两个矩阵
        self.spatial_attention_trend = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
        self.spatial_attention_seasonal = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
        # note 分量变换的矩阵
        # (bs, T, num_nodes, model_dim * 2) -> (bs, T, num_nodes, model_dim*2)
        self.ln = nn.LayerNorm(model_dim * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.adapter = nn.Sequential(
            nn.Linear(model_dim*2, model_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim*4, model_dim*2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DecomposeFormer的中间层：
        1. 同一个sensor，同一个components，先进行时间步方向的自注意力变换，用不同的参数哦
            这一步两部分的输入都是(bs, T, num_nodes, model_dim)，输出也一样
        2. 不同的sensor，同一个components，进行空间方向的自注意力变换，不同参数哦
        3. 不同的components，同一个sensor，同一个时间步，进行
        :param x: 输入的张量，形如 (bs, T, num_nodes, 2, model_dim)
        :return: 输出的张量，形如(bs, T, num_nodes, 2, model_dim)
        """
        bs, T, num_nodes, _, model_dim = x.shape
        x_trend = x[:, :, :, 0, :]
        x_seasonal = x[:, :, :, 1, :]
        # note 进行趋势部分的时序，空域变换
        # (bs, T, num_nodes, model_dim) -> (bs, T, num_nodes, model_dim)
        x_trend_out = self.spatial_attention_trend(x_trend, dim=2)
        # note 进行季节部分的时序，空域变换
        # (bs, num_nodes, T, model_dim) -> (bs, T, num_nodes, model_dim)
        x_seasonal_out = self.spatial_attention_seasonal(x_seasonal, dim=2)
        # note 进行模态之间的变换
        # (bs, T, num_nodes, model_dim*2)
        x_final = torch.cat([x_trend_out, x_seasonal_out], dim=-1)
        residual = x_final
        x_final = self.adapter(x_final)
        x_final = self.dropout_layer(x_final)
        x_final = self.ln(x_final + residual)
        x_final = x_final.reshape(bs, T, num_nodes, 2, model_dim)
        return x_final

class DecomposeFormer_TTS_BT(nn.Module):
    """
    这个类和STAEformer非常相似，但是需要注意的是，我们先进行3层的时序变换，再进行3层的空间变换，在空间变换的部分进行不同分量之间的交互：
    1. Embedding部分与原版模型保持不变
    2. 先进行3层的时序变换，这一部分和原版的STAEformer是一样的，需要注意的是这里的变换是两个分量分开进行的
    3. 再进行3层的空间变换，这一部分和adv模型是一样的，先分别进行空间变换吗，再进行合并
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
            kernel: int = 3,
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
        self.trend_embedding = Embedder(num_nodes, in_steps, input_dim, input_embedding_dim, steps_per_day,
                                        tod_embedding_dim,
                                        dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim)
        self.seasonal_embedding = Embedder(num_nodes, in_steps, input_dim, input_embedding_dim, steps_per_day,
                                           tod_embedding_dim,
                                           dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim)

        # note 回归头
        if use_mixed_proj:
            self.trend_bottleneck = nn.Sequential(
                nn.Linear(in_steps * self.model_dim, self.model_dim * 4),
                nn.ReLU()
            )
            self.output_proj_trend = nn.Linear(self.model_dim * 4, out_steps * output_dim)
            self.seasonal_bottleneck = nn.Sequential(
                nn.Linear(in_steps * self.model_dim, self.model_dim * 4),
                nn.ReLU()
            )
            self.output_proj_seasonal = nn.Linear(4 * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim * 2, self.output_dim)
        print("Kernel Size", kernel)
        self.decomposer = moving_avg(kernel_size=kernel, stride=1)
        # note 先进行3层的时序变换
        self.trend_t_attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.seasonal_t_attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        # note 再进行3层的空间变换
        self.s_attn_layers = nn.ModuleList(
            [
                DecomposeFormer_TTS_layer(self.model_dim, feed_forward_dim, num_heads, dropout)
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
        #print("before deep", x.shape)
        # note 时序变换 (batch_size, in_steps, num_nodes, model_dim)
        for i in range(self.num_layers):
            trend_embeddings = self.trend_t_attn_layers[i](trend_embeddings, dim=1)
            seasonal_embeddings = self.seasonal_t_attn_layers[i](seasonal_embeddings, dim=1)
        # (bs, in_steps, num_nodes, 2, model_dim)
        x = torch.stack([trend_embeddings, seasonal_embeddings], dim=-2)
        # note 空间变换
        # (bs, in_steps, num_nodes, 2, model_dim)
        for attn in self.s_attn_layers:
            x = attn(x)

        # (batch_size, in_steps, num_nodes, 2, model_dim)
        # note 整理一下需要输出的表示
        # (batch_size, num_nodes, in_steps, 2, model_dim)
        out = x.transpose(1, 2)
        # note 得到趋势部分的预测
        out_trend = out[:, :, :, 0, :].reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        # (batch_size, num_nodes, 4 * model_dim)
        pred_trend_repr = self.trend_bottleneck(out_trend).view(
            batch_size, self.num_nodes, 4 * self.model_dim
        )
        pred_trend = self.output_proj_trend(pred_trend_repr).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        pred_trend = pred_trend.transpose(1, 2)
        # note 得到季节部分的预测
        out_seasonal = out[:, :, :, 1, :].reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        pred_seasonal_repr = self.seasonal_bottleneck(out_seasonal).view(batch_size, self.num_nodes, 4 * self.model_dim)
        pred_seasonal = self.output_proj_seasonal(pred_seasonal_repr).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        pred_seasonal = pred_seasonal.transpose(1, 2)
        out = pred_trend + pred_seasonal
        future_trend = self.decomposer(future_data[:, :, :, 0]).unsqueeze(-1)
        # note 整理一下表示
        # (bs, num_nodes, 8*model_dim)
        repr = torch.cat([pred_trend_repr, pred_seasonal_repr], dim=-1)
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

class DecomposeFormer_TTS_TBT(nn.Module):
    """
    这个类和STAEformer非常相似，但是需要注意的是，我们先进行3层的时序变换，再进行3层的空间变换，在空间变换的部分进行不同分量之间的交互：
    1. Embedding部分与原版模型保持不变
    2. 先进行3层的时序变换，这一部分和原版的STAEformer是一样的，需要注意的是这里的变换是两个分量分开进行的
    3. 再进行3层的空间变换，这一部分和adv模型是一样的，先分别进行空间变换吗，再进行合并
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
            kernel: int = 3,
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
        self.trend_embedding = Embedder(num_nodes, in_steps, input_dim, input_embedding_dim, steps_per_day,
                                        tod_embedding_dim,
                                        dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim)
        self.seasonal_embedding = Embedder(num_nodes, in_steps, input_dim, input_embedding_dim, steps_per_day,
                                           tod_embedding_dim,
                                           dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim)

        # note 回归头
        if use_mixed_proj:
            self.trend_bottleneck = nn.Sequential(
                nn.Linear(in_steps * self.model_dim, self.model_dim * 4),
                nn.ReLU()
            )
            self.output_proj_trend = nn.Linear(self.model_dim * 4, out_steps * output_dim)
            self.output_proj_seasonal = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim * 2, self.output_dim)
        print("Kernel Size", kernel)
        self.decomposer = moving_avg(kernel_size=kernel, stride=1)
        # note 先进行3层的时序变换
        self.trend_t_attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.seasonal_t_attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        # note 再进行3层的空间变换
        self.s_attn_layers = nn.ModuleList(
            [
                DecomposeFormer_TTS_layer(self.model_dim, feed_forward_dim, num_heads, dropout)
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
        #print("before deep", x.shape)
        # note 时序变换 (batch_size, in_steps, num_nodes, model_dim)
        for i in range(self.num_layers):
            trend_embeddings = self.trend_t_attn_layers[i](trend_embeddings, dim=1)
            seasonal_embeddings = self.seasonal_t_attn_layers[i](seasonal_embeddings, dim=1)
        # (bs, in_steps, num_nodes, 2, model_dim)
        x = torch.stack([trend_embeddings, seasonal_embeddings], dim=-2)
        # note 空间变换
        # (bs, in_steps, num_nodes, 2, model_dim)
        for attn in self.s_attn_layers:
            x = attn(x)

        # (batch_size, in_steps, num_nodes, 2, model_dim)
        # note 整理一下需要输出的表示
        # (batch_size, num_nodes, in_steps, 2, model_dim)
        out = x.transpose(1, 2)
        # note 得到趋势部分的预测
        out_trend = out[:, :, :, 0, :].reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        # (batch_size, num_nodes, 4 * model_dim)
        pred_trend_repr = self.trend_bottleneck(out_trend).view(
            batch_size, self.num_nodes, 4 * self.model_dim
        )
        pred_trend = self.output_proj_trend(pred_trend_repr).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        pred_trend = pred_trend.transpose(1, 2)
        # note 得到季节部分的预测
        out_seasonal = out[:, :, :, 1, :].reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        pred_seasonal = self.output_proj_seasonal(out_seasonal).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        pred_seasonal = pred_seasonal.transpose(1, 2)
        out = pred_trend + pred_seasonal
        future_trend = self.decomposer(future_data[:, :, :, 0]).unsqueeze(-1)
        # note 整理一下表示
        # (bs, num_nodes, 8*model_dim)
        repr = torch.cat([pred_trend_repr, out_seasonal], dim=-1)
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
