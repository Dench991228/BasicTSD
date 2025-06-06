from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from baselines.SSL.arch.PatchPerm import PatchPermAugmentation
from baselines.SSL.arch.SoftCLT_loss import SoftCLT_Loss


class nconv(nn.Module):
    """Graph conv operation."""

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # (B, D, N, T), (N, N) ->
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    """Linear layer."""

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    """Graph convolution network."""

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # (B, dilation, N, L)
        # note token mixing
        out = [x]
        for a in support:
            x1 = self.nconv(x, a.to(x.device))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a.to(x.device))
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GraphWaveNet(nn.Module):
    """
    Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
    Link: https://arxiv.org/abs/1906.00121
    Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    Venue: IJCAI 2019
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, num_nodes, dropout=0.3, supports=None,
                 gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, dilation: int = 2,
                 ssl_name: str = None, ssl_loss_weight: float = None, **kwargs
                 ):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= dilation
                receptive_field += additional_scope
                additional_scope *= dilation
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        # note 设置和自监督损失有关的内容
        self.ssl_metric_name = ssl_name
        self.ssl_loss_weight = ssl_loss_weight
        #if ssl_name is not None:
            #if ssl_name == "softclt":
                #self.ssl_module = SoftCLT_Loss(similarity_metric="mse", alpha=kwargs["alpha"],tau=kwargs["tau"],
                                               #hard=kwargs["hard"],)
                #self.ssl_module.ssl_loss_weight = ssl_loss_weight
                #self.ssl_module.ssl_metric_name = ssl_name
                #self.ppa_aug = PatchPermAugmentation()
            #else:
                #self.ssl_module = SoftCLT_Loss(similarity_metric="mse", alpha=kwargs["alpha"],tau=kwargs["tau"],
                                               #hard=kwargs["hard"])

    def _forward(self, history_data: torch.Tensor,
                future_data: torch.Tensor,
                batch_seen: int,
                epoch: int,
                train: bool,
                return_repr: bool = False,
                **kwargs) -> torch.Tensor|Dict:
        """Feedforward function of Graph WaveNet.

        Args:
            history_data (torch.Tensor): shape [B, L, N, C]

        Returns:
            torch.Tensor: [B, L, N, 1]
        """
        # (B, L, N, C) -> (B, C, N, L)
        input = history_data.transpose(1, 3).contiguous()
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(
                input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        # (B, C, N, L) -> (B, residual, N, L)
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            # (B, residual, N, L) -> (B, dilation, N, L)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)
        # (B, dim, nodes, T) -> (B, nodes, dim)
        repr = skip[:, :, :, 0].permute((0, 2, 1)).contiguous()
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        # (B, dim, nodes, T) -> (B, nodes, dim)
        x = self.end_conv_2(x)
        if not return_repr:
            return x
        else:
            return {
                "prediction": x,
                "representation": repr,
            }

    def forward(self, history_data: torch.Tensor,
                future_data: torch.Tensor,
                batch_seen: int,
                epoch: int,
                train: bool,
                return_repr: bool = False,
                **kwargs) -> torch.Tensor:
        # note 如果没有对比损失的话
        # prediction: 预测结果，repr (B, node, dim)
        original_forward_output =  self._forward(history_data, future_data, batch_seen, epoch, train, return_repr=True)
        # note 考虑和对比损失有关的内容
        #if self.ssl_metric_name is not None:
            # 先数据增强, 先交换，再增加高斯噪声
            #augmented_data = self.ppa_aug(history_data)
            #rand_noise = torch.randn_like(augmented_data).cuda()*0.01
            #augmented_data += rand_noise
            # 之后再收集对比学习部分的损失
            #augmented_forward_output = self._forward(augmented_data, future_data, batch_seen, epoch, train, return_repr=True)
            #print(original_forward_output["representation"].shape, augmented_forward_output["representation"].shape)
            #ssl_loss = self.ssl_module(original_forward_output["representation"],
                            #augmented_forward_output["representation"],
                            #history_data[:, :, :, 0],
                            #augmented_data[:, :, :, 0]
                            #)
            #original_forward_output['other_losses'] = ssl_loss
        return original_forward_output