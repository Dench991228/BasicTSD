from typing import Dict

import torch
from torch import nn as nn

from baselines.MyModel.arch.PatchFormer import SelfAttentionLayer


class PatchFormer_base(nn.Module):
    """
    Paper: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
    Link: https://arxiv.org/abs/2308.10425
    Official Code: https://github.com/XDZhelheim/STAEformer
    Venue: CIKM 2023
    Task: Spatial-Temporal Forecasting
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
        patch_size: int = 6,
        stride: int = 3,
        hidden_dim: int = 128,
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
        self.hidden_dim = hidden_dim
        # note 嵌入的部分
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
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
        # note patching的部分
        self.patch_size = patch_size
        self.stride = stride
        self.patching_layer = nn.Linear(self.patch_size * self.model_dim, self.hidden_dim)
        self.count_patches = (in_steps - patch_size) // stride + 1
        # note 回归头
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                self.count_patches * self.hidden_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int,
                epoch: int, train: bool, return_repr: bool = False, **kwargs) -> torch.Tensor|Dict:
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
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
        # note 最终的嵌入层输出
        # (batch_size, in_steps, num_nodes, model_dim)
        x = torch.cat(features, dim=-1)
        # note 这里进行patching
        # (batch_size, num_nodes, in_steps, model_dim) -> (bs, num_nodes, count_p, p_size, model_dim)
        x = torch.transpose(x, 1, 2)
        x = x.unfold(-2, self.patch_size, self.stride)
        bs, nodes, count_p, p_size, model_dim = x.shape
        x = x.reshape(bs, nodes, count_p, -1)
        # (bs, num_nodes, count_p, hidden_dim)
        x = self.patching_layer(x)
        # (bs, count_p, num_nodes, hidden_dim)
        x = x.transpose(1, 2).contiguous()
        # note 进行自注意力变换
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        # (batch_size, in_steps, num_nodes, hidden_dim)
        # note 这里才是需要输出的地方
        # (batch_size, num_nodes, in_steps, hidden_dim)
        repr = x.transpose(1, 2)
        repr = repr.reshape(batch_size, -1, self.hidden_dim * self.count_patches)
        # print(repr.shape)
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, hidden_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.hidden_dim * self.count_patches
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)
        if not return_repr:
            return out
        else:
            return {
                "prediction": out,
                "representation": repr,
            }
