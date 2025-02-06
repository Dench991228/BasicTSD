import numpy as np
import torch
from .mae import masked_mae


def masked_trend_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    这个函数和真正的MAE基本上是一样的，但是如果输入为[I, T, N, C]的话，我们会计算每一个像元预测窗口的均值和真值的均值的差值。如果是[I, N, C]的话，退化为普通的MAE
    important 这个函数的目的，是为了验证频率域系列方法是不是真的因为对总体趋势把握不好，导致的性能下降
    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor.
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute error.

    """
    # note 如果是三维的
    if len(prediction.shape) == 3:
        return masked_mae(prediction, target, null_val=null_val)

    # note 如果是四维的，计算均值
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero
    # note 先计算平均值
    # [I, T, N, C] -> [I, N, C]
    prediction_t_mean = torch.mean(prediction * mask, dim=1)
    target_t_mean = torch.mean(target * mask, dim=1)
    loss = torch.abs(prediction_t_mean - target_t_mean)
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)
