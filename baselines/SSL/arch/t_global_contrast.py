import torch.nn as nn
import torch

class TGlobalContrast(nn.Module):
    """
    这个函数的本质，其实是一种local-global contrast， global的部分是全局表示的平均值
    """
    def __init__(self, repr_dim, **kwargs):
        super(TGlobalContrast, self).__init__()
        # note 用来做线性变换的地方
        self.net = nn.Linear(repr_dim, repr_dim)
        self.sigmoid = nn.Sigmoid()
        self.b_xent = nn.BCEWithLogitsLoss()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, model_reprs: torch.Tensor):
        """
        对输入的特征进行t-contrast
        :param model_reprs: 形状是(bs, num_nodes, model_dim)
        """
        bs, num_nodes, model_dim = model_reprs.shape
        # note 首先先计算出context
        # (bs, num_nodes, model_dim) -> (bs, model_dim) -> (bs, 1, model_dim)
        context = self.sigmoid(torch.mean(model_reprs, dim=1)).unsqueeze(1)
        context = context.expand_as(model_reprs).contiguous()
        print("Context Shape: ", context.shape)
        # note 重新映射一下正样本
        model_reprs = self.net(model_reprs)
        # note 得到负样本
        negative_sample_idx = torch.randperm(bs, device=model_reprs.device)
        negative_reprs = model_reprs[negative_sample_idx]
        # (bs, num_nodes)
        # note 分别得到正负样本的分数
        sc_p = torch.nn.functional.cosine_similarity(model_reprs, context, dim=-1)
        sc_n = torch.nn.functional.cosine_similarity(negative_reprs, context, dim=-1)
        # note 计算对比损失
        # (bs, 2 * num_nodes)
        logits = torch.cat([sc_p, sc_n], dim=1)
        print("Logit Shape: ", logits.shape)
        label_p = torch.ones(bs, num_nodes, device=model_reprs.device, dtype=model_reprs.dtype)
        label_n = torch.zeros(bs, num_nodes, device=model_reprs.device, dtype=model_reprs.dtype)
        labels = torch.cat([label_p, label_n], dim=1)
        loss = self.b_xent(logits, labels)
        return loss

class TGlobalContrast_better(nn.Module):
    def __init__(self, repr_dim, hidden_dim=128, tau=0.2,**kwargs):
        super(TGlobalContrast_better, self).__init__()
        # note 用来做线性变换的地方
        self.net1 = nn.Linear(repr_dim, hidden_dim)
        self.net2 = nn.Linear(repr_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.criteria = nn.CrossEntropyLoss()
        for m in self.modules():
            self.weights_init(m)
        self.hidden_dim = hidden_dim
        self.tau = tau
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, model_reprs: torch.Tensor):
        """
        对输入的特征进行t-contrast
        :param model_reprs: 形状是(bs, num_nodes, model_dim)
        """
        bs, num_nodes, model_dim = model_reprs.shape
        # note 首先先计算出context
        # (bs, num_nodes, model_dim) -> (bs, model_dim)
        context = self.net2(torch.mean(model_reprs, dim=1))
        context = torch.nn.functional.normalize(context, dim=-1)
        print("Context Shape: ", context.shape)
        # note 重新映射一下正样本
        model_reprs = self.net1(model_reprs).reshape(bs*num_nodes, -1)
        model_reprs = torch.nn.functional.normalize(model_reprs, dim=-1)
        # note 得到正样本和Context之间的两两关系
        # (bs*num_nodes, bs)
        score = torch.matmul(model_reprs, context.T) / self.tau
        # note 计算对比损失
        # (bs) -> (bs, 1) -> (bs, num_nodes)
        labels = torch.arange(bs, device=model_reprs.device).unsqueeze(-1).repeat(1, num_nodes)
        # (bs, num_nodes)
        labels = labels.reshape(bs * num_nodes)
        loss = self.criteria(score, labels)
        return loss
