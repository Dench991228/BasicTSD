import torch
import numpy as np

from basicts.metrics import masked_mae


def decomposeFormerLoss(prediction: torch.Tensor, target: torch.Tensor, null_val, trend_prediction: torch.Tensor, trend_label: torch.Tensor, **kwargs) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = masked_mae(prediction, target, null_val)
    if trend_prediction is not None:
        t_loss = masked_mae(trend_prediction, trend_label, null_val)
        loss += t_loss
    return loss
