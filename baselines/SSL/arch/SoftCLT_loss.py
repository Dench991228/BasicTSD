# 这个文件的目的，是计算SoftCLT的损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCLT_Loss(nn.Module):
    def __init__(self, similarity_metric: str, alpha: float, tau: float, hard: bool = False, **kwargs):
        """
        SoftCLT损失，来自于(24-ICLR) Soft Contrastive Learning for Time Series，这个损失的意义是让编码出来的时间序列表示之间的距离能够反映定义在数据空间的DTW距离
        :param similarity_metric: {mse, correntropy, correlation, cosine}
        :param alpha: 两个不同时间序列之间权重的上限
        :param tau: 用来调控软正例权重的参数，控制尖锐程度
        """
        super(SoftCLT_Loss, self).__init__()
        self.similarity_metric_name = similarity_metric
        self.alpha = alpha
        self.tau = tau
        self.hard = hard


    def forward(self,
                original_feature: torch.Tensor,
                augmented_feature: torch.Tensor,
                original_series: torch.Tensor,
                augmented_series: torch.Tensor
                ) -> torch.Tensor:
        """
        输入的内容包括：
        :param original_feature: 原始输入数据的特征, 形状为(B, N, D)
        :param augmented_feature: 数据增强后，编码的特征，形状为(B, N, D)
        :param original_series: 输入的原始时间序列，用来计算距离，形状为(B, T, N)
        :param augmented_series: 输入的增强时间序列(B, T, N)
        :return softclt损失
        """
        # note 整理一下原始序列
        b, t, n = original_series.shape
        original_series = original_series.reshape(b, n, t).contiguous()
        augmented_series = augmented_series.reshape(b, n, t).contiguous()

        # note 整理一下特征
        batch_size, nodes, dims = original_feature.shape
        _, _, T = original_series.shape
        # (B, N, D) -> (B*N, D)
        original_feature = original_feature.reshape(-1, dims)
        augmented_feature = augmented_feature.reshape(-1, dims)

        # note 计算两两特征之间的距离
        # (2*B*N, D)
        all_feature = torch.cat((original_feature, augmented_feature), dim=0)
        # (2*B*N, 2*B*N), 直接得到两两相似度
        sim = torch.matmul(all_feature, all_feature.transpose(0, 1))
        print("sim", sim.shape)
        # (2*B*N, 2*B*N-1)
        logits = torch.tril(sim, diagonal=-1)[:, :-1]
        logits +=torch.triu(sim, diagonal=-1)[:, 1:]

        # note 计算两两特征之间的对比损失
        # (2*b*N, 2*B*N-1)
        logits = -F.log_softmax(logits, dim=-1)

        # note 计算两两特征之间的权重，基于输入序列
        # (B*N, T) -> (2*B*N, T)
        original_series = original_series.reshape(-1, T)
        augmented_series = augmented_series.reshape(-1, T)
        # (2*B*N, 2*B*N)
        if not self.hard: # 如果是软对比学习，需要增加对原始数据的计算
            # (B, B)
            pairwise_dist = self.pairwise_dist_mse(original_series).repeat((2, 2))
            soft_label = 2 * self.alpha*F.sigmoid(-self.tau * pairwise_dist)
            # 接下来去掉对角线 (2*B*N, 2*B*N-1)
            soft_label_no_diag = torch.tril(soft_label, diagonal=-1)[:, :-1]
            soft_label_no_diag += torch.triu(soft_label, diagonal=-1)[:, 1:]

            # note 得到最终的对比损失
            loss = logits * soft_label_no_diag
            loss = torch.mean(loss)
            # print(torch.mean(soft_label), loss)
        else: # 如果是硬对比学习，就和自己对比即可
            # note 这一部分是原始样本和增强样本的对比
            i = torch.arange(batch_size, device=original_series.device)
            loss1 = torch.mean(logits[i, batch_size + i - 1])
            loss2 = torch.mean(logits[batch_size + i, i])
            loss = (loss1 + loss2) / 2
            # print(loss)
        return loss

    def pairwise_dist_mse(self, all_features: torch.Tensor):
        """
        计算两两之间的软标签
        note 直接依靠原始序列进行计算，计算完了之后得到(B, B)的相似度矩阵，之后再进行
        :param all_features: 包含Batch个样本，其中每一个样本有d维的张量，计算两两之间的相似性
        :return 形状为(B, B)的矩阵，代表了两两之间的mse
        """
        # (B, d) -> (B), 每一行是这一列的全部平方和
        sum_sq = all_features.pow(2).sum(dim=1)
        # (B, d) * (d, B) 每一行都是两个向量对位相乘之和
        dot_product = torch.mm(all_features, all_features.transpose(-1, -2))
        # (B, 1) + (1, B) - 2*(B, B)
        mse_matrix = (sum_sq.unsqueeze(-1) + sum_sq.unsqueeze(0) - 2 * dot_product)
        return mse_matrix

class SoftCLT_Loss_Sub(SoftCLT_Loss):
    def __init__(self, similarity_metric: str, alpha: float, tau: float, ssl_input_dim: int, hard: bool = False, **kwargs):
        super().__init__(similarity_metric, alpha, tau, hard, **kwargs)
        # note 增加一个用来映射的部分
        self.mapper = nn.Linear(ssl_input_dim, ssl_input_dim // 2)

    def forward(self,
                original_feature: torch.Tensor,
                augmented_feature: torch.Tensor,
                original_series: torch.Tensor,
                augmented_series: torch.Tensor
                ) -> torch.Tensor:
        original_feature = self.mapper(original_feature)
        augmented_feature = self.mapper(augmented_feature)
        print(original_feature.shape, augmented_feature.shape)
        return SoftCLT_Loss.forward(self, original_feature, augmented_feature, original_series, augmented_series)