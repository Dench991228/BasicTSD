from typing import Dict

import torch.nn as nn
import torch


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
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False, pre_norm = False
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
        self.pre_norm = pre_norm if pre_norm is not None else False

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        if self.pre_norm:
            x = self.ln1(x)  # 先归一化 [[1]]
            x = self.attn(x, x, x)  # 注意力计算 [[3]]
            x = self.dropout1(x)
            x = residual + x  # 残差连接 [[6]]
        # Post-Norm: 先计算注意力，再归一化
        else:
            x = self.attn(x, x, x)
            x = self.dropout1(x)
            x = self.ln1(residual + x)  # 残差连接后归一化 [[6]]

        residual = x
        if self.pre_norm:
            x = self.ln2(x)
            x = self.feed_forward(x)
            x = self.dropout2(x)
            x = residual + x
        else:
            x = self.feed_forward(x)
            x = self.dropout2(x)
            x = self.ln2(residual + x)

        x = x.transpose(dim, -2)
        return x


class DynSCon(nn.Module):
    """
    基于STAEformer和对比学习技术的一个方法
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
        num_layers=None,
        dropout=0.1,
        num_t_layers=None,
        num_s_layers=None,
        pre_norm=False,
        s_pre_norm=False,
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
        # note 回归头
        self.output_proj = nn.Linear(
            in_steps * self.model_dim, out_steps * output_dim
        )
        self.output_proj_middle = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout, pre_norm=pre_norm)
                for _ in range(num_layers if num_layers is not None else num_t_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout, pre_norm=s_pre_norm)
                for _ in range(num_layers if num_layers is not None else num_s_layers)
            ]
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int,
                epoch: int, train: bool, return_repr: bool = False, **kwargs) -> torch.Tensor|Dict:
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
        batch_size, in_steps, num_nodes, variates = x.shape

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

        for attn in self.attn_layers_t: # (batch_size, in_steps, num_nodes, model_dim)
            x = attn(x, dim=1)
        # (batch_size, num_nodes, in_steps, model_dim)
        repr_middle = x.transpose(1, 2)
        repr_middle = repr_middle.reshape(batch_size, num_nodes, -1)
        # (batch-size, num_nodes, in_steps * model_dim)
        output_middle = self.output_proj_middle(repr_middle)
        output_middle = output_middle.view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        output_middle = output_middle.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)
        # note 这里才是需要输出的地方
        # (batch_size, num_nodes, in_steps, model_dim)
        repr = x.transpose(1, 2)
        repr = repr.reshape(batch_size, -1, self.model_dim * self.in_steps)
        # print(repr.shape)
        out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        out = out.reshape(
            batch_size, self.num_nodes, self.in_steps * self.model_dim
        )
        out = self.output_proj(out).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        if not return_repr:
            return {
                "prediction": out,
                "prediction_middle": output_middle,
            }
        else:
            return {
                "prediction": out,
                "representation": repr,
                "prediction_middle": output_middle,
            }

    def get_spatial_embeddings(self) -> torch.Tensor:
        """
        返回当前模型的空间编码，优先是adaptive embedding
        """
        spatial_embeddings = self.adaptive_embedding.clone().detach()
        spatial_embeddings.requires_grad = False
        spatial_embeddings = spatial_embeddings.transpose(0, 1).contiguous()
        spatial_embeddings = spatial_embeddings.reshape(self.num_nodes, -1)
        return spatial_embeddings

    @staticmethod
    def get_dynamic_graph(representations: torch.Tensor, tau: float = 1.0):
        """
        :param representations: 形如(bs, nodes, d)的张量
        :param tau: 温度，用来控制分布的
        :return (bs, nodes, nodes)
        """
        return torch.softmax(torch.bmm(representations, torch.transpose(representations, -1, -2)) / tau, dim=-1)

    @staticmethod
    def get_ego_graph_repr(representations: torch.Tensor, graph: torch.Tensor):
        """
        :param representations: 输入的特征，规整为(bs, nodes, d)
        :param graph: 上面构建的图，形状为(bs, nodes, nodes)
        :return (bs, nodes, d)
        note 后续这里可以调整为top-λ这种
        """
        return torch.bmm(graph, representations)

    @staticmethod
    def get_soft_labels(representations: torch.Tensor, use_aug: bool = True):
        """
        这一步的目的，是根据模型的输入特征，分别得到时序和空间的软标签
        :param representations: 输入特征，如果use_aug=False那就是 (bs, nodes, d)；否则就是(2bs, nodes, d)
        :param use_aug: 是否采用了数据增强
        :return (nodes, bs, bs), (bs, nodes, nodes)
        """
        with torch.no_grad():
            original_reprs = representations if not use_aug else representations[:representations.shape[0]//2]
            # 基于original_reprs计算软标签，如果有数据增强，再扩展一下这个标签
            