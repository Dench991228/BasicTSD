from typing import Dict

import torch.nn as nn
import torch

from baselines.MyModel.arch.MyFormer import moving_avg


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class DecomposeFormer_layer(nn.Module):
    def __init__(self, model_dim: int, feed_forward_dim: int, num_heads=8, dropout=0, mask=False):
        super().__init__()
        # note 空间变换的两个矩阵
        self.spatial_attention_trend = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
        self.spatial_attention_seasonal = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
        # note 时序变换的两个矩阵
        self.temporal_attention_trend = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
        self.temporal_attention_seasonal = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
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
        x_trend_out = self.temporal_attention_trend(x_trend, dim=1)
        # (bs, T, num_nodes, model_dim) -> (bs, T, num_nodes, model_dim)
        x_trend_out = self.spatial_attention_trend(x_trend_out, dim=2)
        # note 进行季节部分的时序，空域变换
        # (bs, T, num_nodes, model_dim)
        x_seasonal_out = self.temporal_attention_seasonal(x_seasonal, dim=1)
        # (bs, num_nodes, T, model_dim) -> (bs, T, num_nodes, model_dim)
        x_seasonal_out = self.spatial_attention_seasonal(x_seasonal_out, dim=2)
        # note 进行模态之间的变换
        # (bs, T, num_nodes, model_dim*2)
        x_final = torch.cat([x_trend_out, x_seasonal_out], dim=-1)
        residual = x_final
        x_final = self.adapter(x_final)
        x_final = self.dropout_layer(x_final)
        x_final = self.ln(x_final + residual)
        x_final = x_final.reshape(bs, T, num_nodes, 2, model_dim)
        return x_final

class Embedder(nn.Module):
    """
    把STAEformer那一堆乱七八糟的东西放进来
    """
    def __init__(self,
                 num_nodes: int,
                 in_steps,
                 input_dim,
                 input_embedding_dim,
                 steps_per_day,
                 tod_embedding_dim,
                 dow_embedding_dim,
                 spatial_embedding_dim,
                 adaptive_embedding_dim):
        super().__init__()
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        # note 嵌入的部分
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day
        if tod_embedding_dim > 0:
            # note (steps per day, d_tod)
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            # note (day per week, d_dow)
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            # note (nodes, d_s)
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            # note (in_steps, nodes, d_a)
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1] * self.steps_per_day
        if self.dow_embedding_dim > 0:
            dow = x[..., 2] * 7
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                tod.long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        return x

class DecomposeFormer(nn.Module):
    """
    像元，分量，时间步
    这个模型仿照MoST的模型设计，依次进行如下的建模：
    1. 同一个分量，同一个像元，先进行时序变换
    2. 同一个分量，同一个像元，空间变换，同一个时间步
    3. 不同的分量，同一个像元，同一个时间步
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
            self.output_proj_seasonal = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
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
        trend_components = self.trend_embedding(trend_components)
        seasonal_components = self.seasonal_embedding(seasonal_components)
        #print("after embedding", trend_components.shape, seasonal_components.shape)
        batch_size = x.shape[0]

        x = torch.stack([trend_components, seasonal_components], dim=-2)
        #print("before deep", x.shape)
        # note 花式变换 (batch_size, in_steps, num_nodes, 2, model_dim)
        for attn in self.attn_layers:
            x = attn(x)
        # (bs, T, num_nodes, 2, model_dim)
        #print("after deep", x.shape)
        # x = x.reshape(batch_size, self.in_steps, self.num_nodes, self.model_dim*2)
        # (batch_size, in_steps, num_nodes, model_dim)
        # note 这里才是需要输出的地方
        # (batch_size, num_nodes, in_steps, model_dim)
        repr = x.transpose(1, 2)
        repr = repr.reshape(batch_size, -1, self.model_dim * self.in_steps * 2)
        #print(repr.shape)
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, 2, model_dim)
            out_trend = self.output_proj(out[:, :, :, 0, :]).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            ).transpose(1, 2)
            out_seasonal = self.output_proj_seasonal(out[:, :, :, 1, :]).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            ).transpose(1, 2)
            out = out_trend + out_seasonal  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)
        future_trend = self.decomposer(future_data[:, :, :, 0]).unsqueeze(-1)
        if not return_repr:
            return {
                "prediction": out,
                "input_trend": trend_components[:, :, :, :1],
                "prediction_trend": out_trend,
                "target_trend": future_trend,
            }
        else:
            return {
                "prediction": out,
                "representation": repr,
                "input_trend": trend_components[:, :, :, :1],
                "prediction_trend": out_trend,
                "target_trend": future_trend,
            }
