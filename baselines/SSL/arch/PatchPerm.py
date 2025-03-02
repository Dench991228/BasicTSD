import torch
from numpy.ma.core import remainder
from torch import nn


def process_sequence(seq: torch.Tensor, block_size=2, p=0.25):
    """
    :param seq: 输入的序列，形如(B, T, N)
    """
    # note 先创建新的输出的位置，并且进行分块
    original_shape = seq.shape
    bs = seq.shape[0]
    seq_length = original_shape[-2]
    count_nodes = seq.shape[-1]

    count_blocks = seq_length // block_size
    blocks = seq.reshape(-1, count_blocks, block_size, count_nodes)
    # (bs(B, N), count_blocks, block_size)
    new_blocks = blocks.clone()
    # 选择并打乱块
    for i in range(bs):
        selected_mask = torch.rand(count_blocks) < p
        selected_indices = torch.where(selected_mask)[0]
        shuffled_indices = selected_indices[torch.randperm(len(selected_indices))]
        new_blocks[i][selected_indices] = blocks[i][shuffled_indices]
    # 合并
    return new_blocks.reshape(original_shape)  # 去除填充部分（如有）

class PatchPermAugmentation(nn.Module):
    """
    对于一个时间序列，将其分解为若干个大小相等且不相交的patch，选中一部分patch，随机交换其顺序
    """
    def __init__(self, selection_rate: float = 0.25, patch_size: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = selection_rate
        self.ps = patch_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        将输入的时间序列进行数据增强
        :param input: 输入的序列，形如(B, T, N, C)
        :return 输出的序列，形如(B, T, N, C)
        """
        series = input[:, :, :, 0]
        remainder = input[:, :, :, 1:]

        return torch.cat([process_sequence(series, self.ps, self.sr).unsqueeze(-1), remainder], dim=-1)