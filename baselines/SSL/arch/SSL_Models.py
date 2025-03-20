from typing import Dict

import torch

from baselines.MyModel.arch.DecomposeFormer import DecomposeFormer
from baselines.MyModel.arch.DecomposeFormer_tts import DecomposeFormer_TTS
from baselines.SSL.arch.PatchPerm import PatchPermAugmentation
from baselines.SSL.arch.SoftCLT_loss import SoftCLT_Loss, SoftCLT_Loss_Sub
from baselines.STAEformer.arch.staeformer_arch import STAEformer
from baselines.STID.arch.stid_arch import STID


class STID_SSL(STID):
    def __init__(self, **kwargs):
        STID.__init__(self, **kwargs)
        ssl_name = kwargs['ssl_name']
        ssl_loss_weight = kwargs['ssl_loss_weight']
        self.ssl_metric_name = ssl_name
        self.ssl_loss_weight = ssl_loss_weight
        if ssl_name is not None:
            if ssl_name == "softclt":
                self.ssl_module = SoftCLT_Loss(**kwargs)
                self.ppa_aug = PatchPermAugmentation()
            elif ssl_name == "softclt_sub":
                print("softclt_sub")
                self.ssl_module = SoftCLT_Loss_Sub(**kwargs)
                self.ppa_aug = PatchPermAugmentation()
            else:
                self.ssl_module = SoftCLT_Loss(similarity_metric="mse", alpha=kwargs["alpha"],tau=kwargs["tau"],
                                               hard=kwargs["hard"])
        self.use_future = kwargs['ssl_use_future'] if 'ssl_use_future' in kwargs else False

    def _forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool, **kwargs) -> torch.Tensor|Dict:
        return STID.forward(self, history_data, future_data, batch_seen, epoch, train, return_repr, **kwargs)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool, **kwargs) -> torch.Tensor|Dict:
        # note 如果没有对比损失的话
        # prediction: 预测结果，repr (B, node, dim)
        original_forward_output =  self._forward(history_data, future_data, batch_seen, epoch, train, return_repr)
        # note 考虑和对比损失有关的内容
        if self.ssl_metric_name is not None:
            # 先数据增强, 先交换，再增加高斯噪声
            augmented_data = self.ppa_aug(history_data)
            rand_noise = torch.randn_like(augmented_data[:, :, :, 0]).cuda()*0.01
            augmented_data[:, :, :, 0] += rand_noise
            # 之后再收集对比学习部分的损失
            augmented_forward_output = self._forward(augmented_data, future_data, batch_seen, epoch, train, return_repr)
            if not self.use_future:
                ssl_loss = self.ssl_module(original_forward_output["representation"],
                                augmented_forward_output["representation"],
                                history_data[:, :, :, 0],
                                augmented_data[:, :, :, 0]
                                )
            else:
                ssl_loss = self.ssl_module(original_forward_output["representation"],
                                augmented_forward_output["representation"],
                                future_data[:, :, :, 0],
                                augmented_data[:, :, :, 0]
                                )

            original_forward_output['other_losses'] = [
                {
                    "weight": self.ssl_loss_weight,
                    "loss": ssl_loss,
                    "loss_name": self.ssl_metric_name
                }
            ]
        return original_forward_output

class STAEformer_SSL(STAEformer):
    def __init__(self, **kwargs):
        STAEformer.__init__(self, **kwargs)
        ssl_name = kwargs['ssl_name']
        ssl_loss_weight = kwargs['ssl_loss_weight']
        self.ssl_metric_name = ssl_name
        self.ssl_loss_weight = ssl_loss_weight
        if ssl_name is not None:
            if ssl_name == "softclt":
                self.ssl_module = SoftCLT_Loss(**kwargs)
                self.ppa_aug = PatchPermAugmentation()
            elif ssl_name == "softclt_sub":
                print("softclt_sub")
                self.ssl_module = SoftCLT_Loss_Sub(**kwargs)
                self.ppa_aug = PatchPermAugmentation()
            else:
                self.ssl_module = SoftCLT_Loss(similarity_metric="mse", alpha=kwargs["alpha"],tau=kwargs["tau"],
                                               hard=kwargs["hard"])
        self.use_future = kwargs['ssl_use_future'] if 'ssl_use_future' in kwargs else False

    def _forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool, **kwargs) -> torch.Tensor|Dict:
        return STAEformer.forward(self, history_data, future_data, batch_seen, epoch, train, return_repr, **kwargs)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool = False, **kwargs) -> torch.Tensor|Dict:
        # note 如果没有对比损失的话
        # prediction: 预测结果，repr (B, node, dim)
        original_forward_output =  self._forward(history_data, future_data, batch_seen, epoch, train, True)
        # note 考虑和对比损失有关的内容
        if self.ssl_metric_name is not None:
            # 先数据增强, 先交换，再增加高斯噪声
            augmented_data = self.ppa_aug(history_data)
            rand_noise = torch.randn_like(augmented_data[:, :, :, 0]).cuda()*0.01
            augmented_data[:, :, :, 0] += rand_noise
            # 之后再收集对比学习部分的损失
            augmented_forward_output = self._forward(augmented_data, future_data, batch_seen, epoch, train, return_repr)
            if not self.use_future:
                ssl_loss = self.ssl_module(original_forward_output["representation"],
                                augmented_forward_output["representation"],
                                history_data[:, :, :, 0],
                                augmented_data[:, :, :, 0]
                                )
            else:
                print("use future!")
                ssl_loss = self.ssl_module(original_forward_output["representation"],
                                augmented_forward_output["representation"],
                                future_data[:, :, :, 0],
                                augmented_data[:, :, :, 0]
                            )
            original_forward_output['other_losses'] = [
                {
                    "weight": self.ssl_loss_weight,
                    "loss": ssl_loss,
                    "loss_name": self.ssl_metric_name
                }
            ]
        if not return_repr:
            del original_forward_output['representation']
        return original_forward_output

class DecomposeFormer_TTS_SSL(DecomposeFormer_TTS):
    def __init__(self, **kwargs):
        DecomposeFormer_TTS.__init__(self, **kwargs)
        ssl_name = kwargs['ssl_name']
        ssl_loss_weight = kwargs['ssl_loss_weight']
        self.ssl_metric_name = ssl_name
        self.ssl_loss_weight = ssl_loss_weight
        if ssl_name is not None:
            if ssl_name == "softclt":
                self.ssl_module = SoftCLT_Loss(**kwargs)
                self.ppa_aug = PatchPermAugmentation()
            elif ssl_name == "softclt_sub":
                print("softclt_sub")
                self.ssl_module = SoftCLT_Loss_Sub(**kwargs)
                self.ppa_aug = PatchPermAugmentation()
            else:
                self.ssl_module = SoftCLT_Loss(similarity_metric="mse", alpha=kwargs["alpha"],tau=kwargs["tau"],
                                               hard=kwargs["hard"])
        self.use_future = kwargs['ssl_use_future'] if 'ssl_use_future' in kwargs else False

    def _forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool, **kwargs) -> torch.Tensor|Dict:
        return DecomposeFormer_TTS.forward(self, history_data, future_data, batch_seen, epoch, train, True, **kwargs)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool = False, **kwargs) -> torch.Tensor|Dict:
        bs, in_steps, num_nodes, _ = history_data.shape
        # note 如果没有对比损失的话
        # prediction: 预测结果，repr (B, node, dim)
        original_forward_output =  self._forward(history_data, future_data, batch_seen, epoch, train, return_repr)
        # (bs, num_node, in_steps * 2 * model_dim)
        original_repr = original_forward_output["representation"].reshape(bs, num_nodes, in_steps, 2, self.model_dim)
        # (bs, num_node, in_step * model_dim)
        original_repr_trend = original_repr[:, :, :, 0, :].reshape(bs, num_nodes, -1)
        # note 考虑和对比损失有关的内容
        if self.ssl_metric_name is not None:
            # 先数据增强, 先交换，再增加高斯噪声
            augmented_data = self.ppa_aug(history_data)
            rand_noise = torch.randn_like(augmented_data[:, :, :, 0]).cuda()*0.01
            augmented_data[:, :, :, 0] += rand_noise
            # 之后再收集对比学习部分的损失
            augmented_forward_output = self._forward(augmented_data, future_data, batch_seen, epoch, train, return_repr)
            # (bs, num_node, in_steps * 2 * model_dim)
            augmented_repr = augmented_forward_output["representation"].reshape(bs, num_nodes, in_steps, 2,
                                                                              self.model_dim)
            # (bs, num_node, in_step * model_dim)
            augmented_repr_trend = augmented_repr[:, :, :, 0, :].reshape(bs, num_nodes, -1)

            if not self.use_future:
                ssl_loss = self.ssl_module(original_repr_trend,
                                augmented_repr_trend,
                                original_forward_output['input_trend'][:, :, :, 0],
                                augmented_data[:, :, :, 0]
                                )
            else:
                print("use future!")
                ssl_loss = self.ssl_module(original_repr_trend,
                                augmented_repr_trend,
                                future_data[:, :, :, 0],
                                augmented_data[:, :, :, 0]
                            )
            original_forward_output['other_losses'] = [
                {
                    "weight": self.ssl_loss_weight,
                    "loss": ssl_loss,
                    "loss_name": self.ssl_metric_name
                }
            ]
        if not return_repr:
            del original_forward_output['representation']
        return original_forward_output

class DecomposeFormer_TTS_S_SSL(DecomposeFormer_TTS):
    def __init__(self, **kwargs):
        DecomposeFormer_TTS.__init__(self, **kwargs)
        ssl_name = kwargs['ssl_name']
        ssl_loss_weight = kwargs['ssl_loss_weight']
        self.ssl_metric_name = ssl_name
        self.ssl_loss_weight = ssl_loss_weight
        if ssl_name is not None:
            if ssl_name == "softclt":
                self.ssl_module = SoftCLT_Loss(**kwargs)
                self.ppa_aug = PatchPermAugmentation()
            elif ssl_name == "softclt_sub":
                print("softclt_sub")
                self.ssl_module = SoftCLT_Loss_Sub(**kwargs)
                self.ppa_aug = PatchPermAugmentation()
            else:
                self.ssl_module = SoftCLT_Loss(similarity_metric="mse", alpha=kwargs["alpha"],tau=kwargs["tau"],
                                               hard=kwargs["hard"])
        self.use_future = kwargs['ssl_use_future'] if 'ssl_use_future' in kwargs else False

    def _forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool, **kwargs) -> torch.Tensor|Dict:
        return DecomposeFormer_TTS.forward(self, history_data, future_data, batch_seen, epoch, train, True, **kwargs)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                return_repr: bool = False, **kwargs) -> torch.Tensor|Dict:
        bs, in_steps, num_nodes, _ = history_data.shape
        # note 如果没有对比损失的话
        # prediction: 预测结果，repr (B, node, dim)
        original_forward_output =  self._forward(history_data, future_data, batch_seen, epoch, train, return_repr)
        # (bs, num_node, in_steps * 2 * model_dim)
        original_repr = original_forward_output["representation"].reshape(bs, num_nodes, in_steps, 2, self.model_dim)
        # (bs, num_node, in_step * model_dim)
        original_repr_trend = original_repr[:, :, :, 1, :].reshape(bs, num_nodes, -1)
        # note 考虑和对比损失有关的内容
        if self.ssl_metric_name is not None:
            # 先数据增强, 先交换，再增加高斯噪声
            augmented_data = self.ppa_aug(history_data)
            rand_noise = torch.randn_like(augmented_data[:, :, :, 0]).cuda()*0.01
            augmented_data[:, :, :, 0] += rand_noise
            # 之后再收集对比学习部分的损失
            augmented_forward_output = self._forward(augmented_data, future_data, batch_seen, epoch, train, return_repr)
            # (bs, num_node, in_steps * 2 * model_dim)
            augmented_repr = augmented_forward_output["representation"].reshape(bs, num_nodes, in_steps, 2,
                                                                              self.model_dim)
            # (bs, num_node, in_step * model_dim)
            augmented_repr_trend = augmented_repr[:, :, :, 1, :].reshape(bs, num_nodes, -1)

            if not self.use_future:
                ssl_loss = self.ssl_module(original_repr_trend,
                                augmented_repr_trend,
                                history_data[:, :, :, 0] - original_forward_output['input_trend'][:, :, :, 0],
                                augmented_data[:, :, :, 0]
                                )
            else:
                print("use future!")
                ssl_loss = self.ssl_module(original_repr_trend,
                                augmented_repr_trend,
                                future_data[:, :, :, 0],
                                augmented_data[:, :, :, 0]
                            )
            original_forward_output['other_losses'] = [
                {
                    "weight": self.ssl_loss_weight,
                    "loss": ssl_loss,
                    "loss_name": self.ssl_metric_name
                }
            ]
        if not return_repr:
            del original_forward_output['representation']
        return original_forward_output

