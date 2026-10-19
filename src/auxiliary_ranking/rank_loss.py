
import torch
import operator

from typing import Tuple
from dataclasses import dataclass
from torch.nn import functional as F
from chemprop.nn import metrics, BoundedMSE
from lightning.pytorch.core.mixins import HyperparametersMixin

import logging

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


@dataclass(frozen=True)
class Cond:
    target: int
    op: str                      # "<", "<=", ">", ">=", "isnan", "notnan"
    value: float = float("nan")  # unused for isnan / notnan
    or_nan: bool = False         # if True, a missing value also satisfies the condition

@dataclass(frozen=True)
class CrossRule:
    winner: Tuple[Cond, ...] 
    loser: Tuple[Cond, ...]  

@dataclass(frozen=True)
class Level:
    target: int
    margin: float
    higher_wins: bool = True

@dataclass(frozen=True)
class RankRuleConfig:
    levels: Tuple[Level, ...] = ()          # priority order, first = highest
    cross_rules: Tuple[CrossRule, ...] = ()
    hierarchical: bool = True               # True: first level where both samples have a value decides
                                            # False: every level votes independently, any "too close" vetoes
    cross_rules_respect_veto: bool = False

def _group_flag(T, conds):
    flag = torch.ones(T.size(0), dtype=torch.bool, device=T.device)
    for c in conds:
        col = T[:, c.target]
        nan = torch.isnan(col)
        if c.op == "isnan":
            m = nan
        elif c.op == "notnan":
            m = ~nan
        else:
            m = _OPS[c.op](col, c.value)      # NaN -> False
            if c.or_nan:
                m = m | nan                   # "value satisfies the condition OR is missing"
        flag &= m
    return flag


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
    def __init__(self, cfg):
        super().__init__()
        self.bounded_mse = BoundedMSE()
        self.ce_loss = torch.nn.BCEWithLogitsLoss(reduce='mean')
        self.alias = 'combined_regression_rank_loss'
        self.cfg = cfg
    
    def build_pair_mask(self, T, cfg):
        """T: [B, num_targets] with NaN for missing. Returns [B, B], mask[i, j] = i should outrank j."""
        B = T.size(0)
        dev = T.device
        level_mask = torch.zeros(B, B, dtype=torch.bool, device=dev)
        veto = torch.zeros(B, B, dtype=torch.bool, device=dev)
        undecided = torch.ones(B, B, dtype=torch.bool, device=dev)
    
        for lvl in cfg.levels:                       # loop over levels, not pairs
            v = T[:, lvl.target]
            has = ~torch.isnan(v)
            both = has.unsqueeze(1) & has.unsqueeze(0)
            d = v.unsqueeze(1) - v.unsqueeze(0)
            if not lvl.higher_wins:
                d = -d
    
            active = both & undecided if cfg.hierarchical else both
            level_mask |= active & (d > lvl.margin)
            veto |= active & (d.abs() <= lvl.margin)
            if cfg.hierarchical:
                undecided &= ~both                   # lower levels only see pairs this one couldn't judge
    
        cross = torch.zeros_like(level_mask)
        for r in cfg.cross_rules:                    # loop over rules, not pairs
            w = _group_flag(T, r.winner)
            l = _group_flag(T, r.loser)
            cross |= w.unsqueeze(1) & l.unsqueeze(0)
    
        if cfg.cross_rules_respect_veto:
            cross &= ~veto
    
        mask = (level_mask & ~veto) | cross
        return mask & ~mask.T                        # drop contradictory pairs
    
    def compute_generalized_pair_loss(self, logits_arc, rank_split):
        if logits_arc.dim() > 1 and logits_arc.size(-1) > 1:
            # Logistic difference for binary classification logits
            scores = logits_arc[:, 1] - logits_arc[:, 0]
        else:
            scores = logits_arc.view(-1)
        score_diffs = scores.unsqueeze(1) - scores.unsqueeze(0)
        
        pair_mask = self.build_pair_mask(rank_split, self.cfg)
        if not pair_mask.any():
            return torch.tensor(0.0, device=logits_arc.device, requires_grad=True)
    
        valid_diffs = score_diffs[pair_mask]
        
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
                # with open('loss_log.txt', 'a') as f:
                    # f.write(f'mse_loss: {mse_loss.item():.6f}, pair_loss: {pair_loss_.item():.6f}\n')
                return mse_loss + pair_loss_
        else:
            print(f'issue calculating pair loss')
            return mse_loss