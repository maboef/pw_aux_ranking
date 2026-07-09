
import torch
from chemprop.nn import metrics, BoundedMSE
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch.nn import functional as F

import logging

def ce_loss(logits, targets, reduction="none"):
    """
    cross entropy loss in pytorch.
    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=0)
        if reduction == "none":
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

class MSEPlusPairwiseRankingLoss(metrics.ChempropMetric):
    def __init__(self, rank_dist, categorical_dist):
        super().__init__()
        self.bounded_mse = BoundedMSE()
        self.ce_loss = torch.nn.BCEWithLogitsLoss(reduce='mean')
        self.alias = 'combined_regression_rank_loss'
        self.rank_dist = rank_dist
        self.categorical_dist = categorical_dist
    
    def compute_generalized_pair_loss(self, logits_arc, rank_split):
        if logits_arc.dim() > 1 and logits_arc.size(-1) > 1:
            # Logistic difference for binary classification logits
            scores = logits_arc[:, 1] - logits_arc[:, 0]
        else:
            scores = logits_arc.view(-1)
            
        # --- BROADCASTED PAIRWISE DIFFERENCES ---
        # score_diffs[i, j] = Score_i - Score_j (Shape: [Batch, Batch])
        score_diffs = scores.unsqueeze(1) - scores.unsqueeze(0)
        
        x = rank_split[:, 0]
        y = rank_split[:, 1]
        z = rank_split[:, 2]
        
        has_y = ~torch.isnan(y)
        has_z = ~torch.isnan(z)
        
        # --- BROADCASTED VALUE DIFFERENCES ---
        x_diffs = x.unsqueeze(1) - x.unsqueeze(0)
        y_diffs = y.unsqueeze(1) - y.unsqueeze(0)
        z_diffs = z.unsqueeze(1) - z.unsqueeze(0)
        
        both_have_y = has_y.unsqueeze(1) & has_y.unsqueeze(0)
        both_have_z = has_z.unsqueeze(1) & has_z.unsqueeze(0)
        
        effective_rank_diffs = x_diffs.clone()
        effective_rank_diffs = torch.where(both_have_z, z_diffs, effective_rank_diffs)
        effective_rank_diffs = torch.where(both_have_y, y_diffs, effective_rank_diffs)
        
        # Which rank level is actually being used?
        using_z = both_have_z
        using_y = both_have_y & ~both_have_z
        using_x = ~(using_y | using_z)
        
        pair_mask = (
            (using_x & (x_diffs > self.categorical_dist)) |
            (using_y & (y_diffs > self.rank_dist)) |
            (using_z & (z_diffs > self.rank_dist)))
        if not pair_mask.any():
            return torch.tensor(0.0, device=logits_arc.device, requires_grad=True)
        
        # Extract valid differences
        valid_diffs = score_diffs[pair_mask]
        # --- CROSS ENTROPY LOSS CALCULATION ---
        logits_pair = torch.stack([torch.zeros_like(valid_diffs), valid_diffs], dim=1)
        targets_ranking = torch.tensor([0.0, 1.0], device=valid_diffs.device).expand(len(valid_diffs), 2)
        
        ce_loss = torch.nn.BCEWithLogitsLoss(reduce='mean')
        loss = self.ce_loss(logits_pair, targets_ranking)
        return loss


    def _calc_unreduced_loss(self, preds, targets, logits_mat, logits_arc, 
                             targets_mat, weights=None, lt_mask=None, gt_mask=None,
                             rank_split=None):
        """
        preds:       (batch, tasks) - regression predictions
        targets:     (batch, tasks) - ground truth
        mask:        (batch, tasks) or None
        lt_mask:     (batch, tasks) - less than mask
        gt_mask:     (batch, tasks) - greater than mask
        rank_split:  (batch,) tensor on which the ranking is done, can be categorical or continuous
        """
        mask = targets.isfinite() # all samples that have target value NaN, here the mask will be generated to account for these in loss calculation
        targets = targets.nan_to_num(nan=0.0) # set missing targets to 0.0 as chemprop MSE loss will give error on missing values despite the masking of those values
        mse_loss = self.bounded_mse(preds, targets, mask, weights, lt_mask, gt_mask) # switching back to ungrouped loss due to difficulty converging on valid set while using the cluster splitting
        if rank_split is None:
            return mse_loss
        
        if rank_split is not None:
            pair_loss_ = self.compute_generalized_pair_loss(logits_arc, rank_split)
            
            if torch.isnan(pair_loss_):
                print('pair loss produced NaN')
                return mse_loss
            else:
                # with open('loss_log_.txt', 'a') as f:
                #     f.write(f'mse_loss: {mse_loss.item():.6f}, pair_loss: {pair_loss_.item():.6f}\n')
                return mse_loss + pair_loss_
        else:
            print(f'issue calculating pair loss')
            return mse_loss