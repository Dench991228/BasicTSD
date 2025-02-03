import numpy as np
import torch


def spatial_corr(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    这个函数和corr基本上是一样的，但是需要注意，我们针对每一个时间步计算corr，最终返回全部时间步的平均值

    Args:
        prediction (torch.Tensor): The predicted values as a tensor. (items, T, N, feature)或者(items, N, features)
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor.
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute error.

    """
    # note 基本的操作，和空值这些有关
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    # note 开始按照时间步计算corr
    # 计算N维度平均值
    # (I, T, N, C)或者(I, N, C)
    prediction_mean = torch.mean(prediction, dim=-2, keepdim=True)
    target_mean = torch.mean(target, dim=-2, keepdim=True)

    # 计算偏差 (X - mean_X) 和 (Y - mean_Y)
    prediction_dev = prediction - prediction_mean
    target_dev = target - target_mean

    # 计算皮尔逊相关系数
    # 现在N维度上计算乘积
    numerator = torch.sum(prediction_dev * target_dev, dim=-2, keepdim=True)  # 分子
    # 再在N维度上计算标准差
    denominator = torch.sqrt(torch.sum(prediction_dev ** 2, dim=-2, keepdim=True) * torch.sum(target_dev ** 2, dim=-2, keepdim=True))  # 分母
    loss = numerator / denominator

    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)
