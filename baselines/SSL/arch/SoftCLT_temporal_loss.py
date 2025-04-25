import torch
from torch.nn import functional as F

from baselines.SSL.arch.SoftCLT_loss import SoftCLT_Loss


class SoftCLTLoss_t(SoftCLT_Loss):
    def __init__(self, similarity_metric: str, alpha: float, tau: float, hard: bool = False, normalize: bool = False,
                 tod_adjust: bool = False, **kwargs):
        """
        SoftCLT损失，来自于(24-ICLR) Soft Contrastive Learning for Time Series，这个损失的意义是让编码出来的时间序列表示之间的距离能够反映定义在数据空间的DTW距离
        :param similarity_metric: {mse, correntropy, correlation, cosine}
        :param alpha: 两个不同时间序列之间权重的上限
        :param tau: 用来调控软正例权重的参数，控制尖锐程度
        """
        super().__init__(similarity_metric, alpha, tau, hard, normalize, **kwargs)
        self.tod_adjust = tod_adjust

    def forward(self,
                original_feature: torch.Tensor,
                augmented_feature: torch.Tensor,
                original_series: torch.Tensor,
                augmented_series: torch.Tensor,
                **kwargs):
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
        # (n, b, t)
        original_series = original_series.permute((2, 0, 1)).contiguous()
        print("permuted series shape", original_series.shape)
        # note 整理一下特征
        batch_size, nodes, dims = original_feature.shape
        _, _, T = original_series.shape
        # (B, N, D) -> (N, B, D)
        original_feature = original_feature.transpose(0, 1).contiguous()
        augmented_feature = augmented_feature.transpose(0, 1).contiguous()

        # note 计算两两特征之间的距离
        # 2*(N, B, D) -> (N, 2B, D)
        all_feature = torch.cat((original_feature, augmented_feature), dim=1)
        # (N, 2B, 2B)
        sim = torch.matmul(all_feature, all_feature.transpose(-1, -2))
        print("sim shape", sim.shape)
        # (2*B*N, 2*B*N-1)
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits +=torch.triu(sim, diagonal=-1)[:, :, 1:]

        # note 计算两两特征之间的对比损失
        # (2*b*N, 2*B*N-1)
        logits = -F.log_softmax(logits, dim=-1)
        print("logit shape", logits.shape)
        # note 计算两两特征之间的权重，基于输入序列
        # (2*B*N, 2*B*N)
        if not self.hard: # 如果是软对比学习，需要增加对原始数据的计算
            # (N, B, B)
            pairwise_dist = self.pairwise_dist_mse(original_series).repeat((1, 2, 2))
            print("mean dist", torch.mean(pairwise_dist))
            # note 如果需要考虑tod的话，就在这里加上
            if self.tod_adjust:
                # (B) tod / max_tod
                assert 'tod' in kwargs
                tod = kwargs['tod']*torch.pi*2
                # (B, B) -> (2B, 2B) -> (1, 2B, 2B)
                tod_dist = torch.cos(torch.abs(tod.unsqueeze(-1) - tod.unsqueeze(0))).repeat((2, 2))
                print("tod dist", torch.mean(2-tod_dist))
                pairwise_dist *= (2 - tod_dist.unsqueeze(0))
                print("mean dist after tod", torch.mean(pairwise_dist))
            print("shape of pairwise dist", pairwise_dist.shape)
            soft_label = 2 * self.alpha*F.sigmoid(-self.tau * pairwise_dist)
            # 接下来去掉对角线 (2*B*N, 2*B*N-1)
            soft_label_no_diag = torch.tril(soft_label, diagonal=-1)[:, :, :-1]
            soft_label_no_diag += torch.triu(soft_label, diagonal=-1)[:, :, 1:]

            # note 得到最终的对比损失
            loss = logits * soft_label_no_diag
            loss = torch.mean(loss)
            mean_soft_label = torch.mean(soft_label)
            print(mean_soft_label, loss)
            return [
                {
                    "weight": self.ssl_loss_weight,
                    "loss": loss,
                    "loss_name": self.ssl_metric_name
                }, {
                    "weight": 0.,
                    "loss": mean_soft_label,
                    "loss_name": "mean soft label"
                }
            ]
        else: # 如果是硬对比学习，就和自己对比即可
            # note 这一部分是原始样本和增强样本的对比
            i = torch.arange(batch_size, device=original_series.device)
            loss1 = torch.mean(logits[i, batch_size + i - 1])
            loss2 = torch.mean(logits[batch_size + i, i])
            loss = (loss1 + loss2) / 2
            print(loss)
        return loss

class SoftCLTLoss_global_t(SoftCLT_Loss):
    """
    这个类相比于SoftCLTLoss_t，采用全局的表示之间的相似度来计算时间步距离
    """
    def forward(self,
                original_feature: torch.Tensor,
                augmented_feature: torch.Tensor,
                original_series: torch.Tensor,
                augmented_series: torch.Tensor,
                **kwargs
                ):
        # note 先计算一下全局特征之间的距离
        # (B, D)
        global_repr = torch.mean(original_feature, dim=1)

        # note 首先计算两两特征之间的距离
        # note 整理一下特征
        batch_size, nodes, dims = original_feature.shape
        _, _, T = original_series.shape
        # (B, N, D) -> (N, B, D)
        original_feature = original_feature.transpose(0, 1).contiguous()
        augmented_feature = augmented_feature.transpose(0, 1).contiguous()

        # note 计算两两特征之间的距离
        # 2*(N, B, D) -> (N, 2B, D)
        all_feature = torch.cat((original_feature, augmented_feature), dim=1)
        # (N, 2B, 2B)
        sim = torch.matmul(all_feature, all_feature.transpose(-1, -2))
        print("sim shape", sim.shape)
        # (2*B*N, 2*B*N-1)
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits +=torch.triu(sim, diagonal=-1)[:, :, 1:]

        # note 计算两两特征之间的对比损失
        # (2*b*N, 2*B*N-1)
        logits = -F.log_softmax(logits, dim=-1)
        print("logit shape", logits.shape)

        # note 计算两两特征之间的权重，基于输入序列
        # note 整理一下原始序列
        # (n, b, t)
        original_series = original_series.permute((2, 0, 1)).contiguous()
        print("permuted series shape", original_series.shape)

        # (2*B*N, 2*B*N)
        # (N, B, B)
        time_series_pairwise_dist = self.pairwise_dist_mse(original_series).repeat((1, 2, 2))
        print("mean dist between time series", torch.mean(time_series_pairwise_dist))

        # note 这一步修正原始的序列距离
        normalized_global_repr = F.normalize(global_repr, dim=-1)
        # (B, B) -> (1, B, B) -> (1, 2B, 2B)
        cosine_distance_global_repr = torch.matmul(normalized_global_repr, normalized_global_repr.transpose(-1, -2)).unsqueeze(0).repeat((1, 2, 2))
        time_series_pairwise_dist *= (2 - cosine_distance_global_repr)
        print("mean dist after global repr", torch.mean(time_series_pairwise_dist))
        print("shape of pairwise dist", time_series_pairwise_dist.shape)

        # note 这一步完成软标签计算
        soft_label = 2 * self.alpha*F.sigmoid(-self.tau * time_series_pairwise_dist)
        # 接下来去掉对角线 (2*B*N, 2*B*N-1)
        soft_label_no_diag = torch.tril(soft_label, diagonal=-1)[:, :, :-1]
        soft_label_no_diag += torch.triu(soft_label, diagonal=-1)[:, :, 1:]

        # note 得到最终的对比损失
        loss = logits * soft_label_no_diag
        loss = torch.mean(loss)
        mean_soft_label = torch.mean(soft_label)
        print(mean_soft_label, loss)
        return [
            {
                "weight": self.ssl_loss_weight,
                "loss": loss,
                "loss_name": self.ssl_metric_name
            }, {
                "weight": 0.,
                "loss": mean_soft_label,
                "loss_name": "mean soft label"
            },{
                "weight": 0.,
                "loss": torch.mean(cosine_distance_global_repr),
                "loss_name": "mean global distance"
            }
        ]
