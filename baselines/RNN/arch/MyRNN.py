import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self,
                 count_nodes: int,
                 out_steps: int,
                 in_steps: int,
                 in_features: int = 3,
                 layers: int = 1,
                 hidden_size: int = 64,
                 backbone: str = "lstm", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_arch = backbone
        self.layers = layers
        # note 骨干模型
        if backbone == "lstm":
            self.backbone = nn.LSTM(hidden_size=hidden_size,
                                    input_size=hidden_size,
                                    num_layers=layers,
                                    batch_first=True,
                                    dropout=0.1)
        else:
            self.backbone = nn.GRU(hidden_size=hidden_size,
                                   input_size=hidden_size,
                                   num_layers=layers,
                                   batch_first=True,
                                   dropout=0.1)
        # note 像元嵌入
        self.node_embeddings = nn.Parameter(torch.empty(count_nodes, hidden_size))
        nn.init.xavier_uniform_(self.node_embeddings)
        # note 嵌入层
        self.input_proj = nn.Linear(in_features, hidden_size)
        # note 输出层
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, out_steps),
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        # note 先对输入数据进行嵌入
        # (bs, T, n, f) -> (bs, n, T, f)
        x = history_data.transpose(1, 2)
        # print(x.shape)
        bs, n, t, f = x.shape
        x = x.reshape(bs*n, t, f)
        # print(x.shape)
        # (bs * n, T, d)
        x = self.input_proj(x)
        # (n, d) -> (n, 1, d)
        expanded_embeddings = self.node_embeddings.unsqueeze(-2).unsqueeze(0).repeat(bs, 1, t, 1)
        expanded_embeddings = expanded_embeddings.reshape(x.shape)
        x = x + expanded_embeddings
        # (bs*n, d)
        x = self.backbone(x)[0][:, -1, :]
        x = x.reshape(bs, n, -1)
        # (bs, n, T) -> (bs, n, T, 1) -> (bs, T, n, 1)
        out = self.out_proj(x)
        out = out.unsqueeze(-1)
        out = out.transpose(1, 2)
        return out
