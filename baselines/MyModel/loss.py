import torch
import numpy as np

from basicts.metrics import masked_mae


def NewDecomposeFormerLoss(prediction: torch.Tensor, target: torch.Tensor, null_val, prediction_trend: None, target_trend: None, **kwargs) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = masked_mae(prediction, target, null_val)
    lloss = None
    if prediction_trend is not None:
        lloss = masked_mae(prediction_trend, target_trend, null_val)
    # print(loss, lloss)
    if lloss is not None:
        loss += lloss

    return loss

def DualDecomposeFormerLoss(prediction: torch.Tensor, target: torch.Tensor, null_val, prediction_trend: None, target_trend: None, **kwargs) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = masked_mae(prediction, target, null_val)
    lloss = None
    sloss = None
    if prediction_trend is not None:
        # 长期趋势的loss
        lloss = masked_mae(prediction_trend, target_trend, null_val)
        # 短期波动的loss
        sloss = masked_mae(prediction - prediction_trend, target - target_trend, null_val)
    #print(loss, lloss, sloss)
    if lloss is not None:
        loss += lloss
    if sloss is not None:
        loss += sloss
    return loss
