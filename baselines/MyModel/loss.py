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
