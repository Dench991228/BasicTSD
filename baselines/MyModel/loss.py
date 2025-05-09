import torch
import numpy as np

from basicts.metrics import masked_mae

def double_out_loss(prediction: torch.Tensor, target: torch.Tensor, null_val, prediction_middle: torch.Tensor = None) -> torch.Tensor:
    loss = masked_mae(prediction, target, null_val)
    if prediction_middle is not None:
        lloss = masked_mae(prediction_middle, target, null_val)
        print(loss, lloss)
        loss = (loss + lloss) / 2
    return loss

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
        loss /= 2
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
    scale = 1
    if prediction_trend is not None:
        # 长期趋势的loss
        lloss = masked_mae(prediction_trend, target_trend, null_val)
        # 短期波动的loss
        sloss = masked_mae(prediction - prediction_trend, target - target_trend, null_val)
        scale += 1
        #print(loss, lloss, sloss)
    #print(loss, lloss, sloss)
    if lloss is not None:
        loss += lloss
    if sloss is not None:
        loss += sloss
    loss /= scale
    # print(loss)
    return loss
