import warnings

import torch
from torch import _VF, sym_int as _sym_int, Tensor
from typing import Callable, Optional, TYPE_CHECKING, Union
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)

import numpy as np

def censored_mse_loss(
    pred: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    censoring: Optional[bool] = False,
    weight: Optional = None,
) -> Tensor:
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None) -> Tensor

    Measures the element-wise mean squared error, with optional weighting.

    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        size_average (bool, optional): Deprecated (use reduction).
        reduce (bool, optional): Deprecated (use reduction).
        censoring (bool, optional): use relations from censored data.
        weight (Tensor, optional): Weights for each sample. Default: None.

    Returns:
        Tensor: Mean Squared Error loss (optionally weighted).
    """
    if has_torch_function_variadic(pred, target, weight):
        return handle_torch_function(
            mse_loss,
            (pred, target, weight),
            pred,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            weight=weight,
        )

    if not (target.size() == pred.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({pred.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )

    expanded_input, expanded_target = torch.broadcast_tensors(pred, target)
    device = pred.device

    if censoring:
        # Perform censored MSE loss based on the operators fed in weights
        squared_errors = torch.pow(expanded_input - expanded_target, 2)
        mask = ~torch.isnan(target)

        eq = torch.eq
        lt = torch.lt
        num_eq = (np.array(weight) == eq).sum()
        # relation_mask = ~get_relation_mask_operator_torch(torch.where(mask, target, torch.tensor(0.0, device="cpu")), torch.where(mask, pred, torch.tensor(0.0, device=device)), weight)
        relation_mask = ~get_relation_mask_operator_torch(torch.where(mask, target, torch.tensor(0.0, device=device)), torch.where(mask, pred, torch.tensor(0.0, device=device)), np.where(np.array(mask.cpu()), np.array(weight), eq))
        censored_data_learning = np.array(relation_mask.cpu()).sum() - num_eq
        print(f'num values outside censored range: {censored_data_learning}')
        error = pred - torch.where(relation_mask, target, torch.tensor(1e-8, device=device))
        error *= relation_mask
        squared_errors = error.pow(2)
        return torch.mean(squared_errors), censored_data_learning
        
    else:
        return torch._C._nn.mse_loss(
            expanded_input, expanded_target, _Reduction.get_enum(reduction)
        )

def get_relation_mask_operator_torch(y_true, y_pred, operator):
    """
    Create mask for censored data - mask where y_pred is in censored area will return True - non-censored area will return False
    """
    true_flat = y_true.flatten()
    pred_flat = y_pred.flatten()
    operator = np.array(operator)
    operator_flat = operator.flatten()
    relation_mask = torch.stack([op(a, b) for a, b, op in zip(pred_flat, true_flat, operator_flat)]) # should be op(pred, true)
    relation_mask = relation_mask.reshape(operator.shape)
    return relation_mask