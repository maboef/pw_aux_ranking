
import io
import random
import torch
from torch.amp import autocast
from chemprop.nn import metrics
from chemprop import data, featurizers, models, nn
from lightning.pytorch.core.mixins import HyperparametersMixin

from chemprop.nn.hparams import HasHParams
from chemprop.nn.utils import Activation
from chemprop.nn.predictors import Predictor

from chemprop.schedulers import build_NoamLike_LRSched
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

class Bn_Controller:
    """
    Batch Norm controller
    """

    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.SyncBatchNorm) or isinstance(m, torch.nn.BatchNorm2d):
                self.backup[name + ".running_mean"] = m.running_mean.data.clone()
                self.backup[name + ".running_var"] = m.running_var.data.clone()
                self.backup[name + ".num_batches_tracked"] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.SyncBatchNorm) or isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + ".running_mean"]
                m.running_var.data = self.backup[name + ".running_var"]
                m.num_batches_tracked.data = self.backup[name + ".num_batches_tracked"]
        self.backup = {}


class MultiObjectiveFFN(torch.nn.Module, HasHParams):
    """
    Wrapper around RegressionFFN that adds multi-objective loss capability.
    Instead of wrapping the base_ffn, we create our own FFN structure.
    """
    n_targets = 1
    def __init__(self, input_dim, hidden_dim, n_tasks, n_layers, dropout, activation, 
                 criterion=None, output_transform=torch.nn.Identity):
        super().__init__()
        self.hparams = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'n_tasks': n_tasks,
            'n_layers': n_layers,
            'dropout': dropout,
            'activation': activation,
            'cls': self.__class__,
            'criterion': None,
        }
        
        if criterion is None:
            self.criterion = MSEPlusPairwiseRankingLoss()
        else:
            self.criterion = criterion
        layers = []

        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(torch.nn.LeakyReLU(negative_slope=0.1))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.LeakyReLU(negative_slope=0.1))
        layers.append(torch.nn.Dropout(dropout))
        self.ffn = torch.nn.Sequential(*layers)
        
        # regressor layer
        self.regressor = torch.nn.Linear(hidden_dim, n_tasks)
        
        # ranked classifier layer
        self.arc_classifier = torch.nn.Linear(hidden_dim, 2)
        
        self.output_transform = output_transform()

    def forward(self, x, targets=None):
        """
        Forward pass - returns predictions.
        x: FFN input (batch, input_dim)
        Returns: predictions (batch, n_tasks)
        """
        # Get predictions from FFN
        encoding = self.ffn(x)
        preds = self.regressor(encoding)
        # logits_arc = None
        logits_arc = self.arc_classifier(encoding)
        if targets is None:
            return preds, logits_arc
        # logits_mat, targets_mat = self.compute_rank_logits(logits_arc, targets)
        logits_mat = None
        targets_mat = None
        return self.output_transform(preds), logits_mat, logits_arc, targets_mat

    train_step = forward

    def compute_rank_logits(self, logits, targets=None):
        logits_mat = logits.unsqueeze(dim=0) - logits.unsqueeze(dim=1)
        logits_mat = logits_mat.flatten(0, 1)
        if targets is not None:
            targets_mat = (1 + torch.sign(targets.unsqueeze(dim=0) - targets.unsqueeze(dim=1))) / 2
            targets_mat = targets_mat.flatten(0, 1)
            # one-hot encode the targets_mat
            targets_mat = targets_mat.squeeze()
            targets_onehot = torch.zeros((targets_mat.shape[0], 2)).to(targets_mat.device)
            targets_onehot[:, 0] = targets_mat
            targets_onehot[:, 1] = 1 - targets_mat
            return logits_mat, targets_onehot
        return logits_mat, None

class MSEPlusPairwiseRankingLoss(metrics.ChempropMetric):
    def __init__(self, rank_dist, categorical_dist):
        super().__init__()
        self.bounded_mse = nn.BoundedMSE()
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
                with open('loss_log_.txt', 'a') as f:
                    f.write(f'mse_loss: {mse_loss.item():.6f}, pair_loss: {pair_loss_.item():.6f}\n')
                # print(f'mse + pair: {mse_loss + pair_loss_}')
                return mse_loss + pair_loss_
        else:
            print(f'issue calculating pair loss')
            return mse_loss


class CustomMPNN(models.MPNN):
    """Custom MPNN that handles multi-objective loss with ranking."""

    def training_step(self, batch, batch_idx):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, aux_mask = batch
        Z = self.fingerprint(bmg, V_d, X_d)
        preds, logits_mat, logits_arc, targets_mat = self.predictor.train_step(Z, targets)
        loss = self.criterion._calc_unreduced_loss(preds, targets, logits_mat, logits_arc, 
                                                   targets_mat, weights, lt_mask, gt_mask, aux_mask)
        
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, aux_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds, logits_mat, logits_arc, targets_mat = self.predictor.train_step(Z, targets)
        perf = self.metrics[0](preds, targets, mask, weights, lt_mask, gt_mask)
        # ranked_perf = self.criterion._calc_unreduced_loss(preds, targets, logits_mat, logits_arc, targets_mat, weights, lt_mask, gt_mask, aux_mask)
        
        # print(f'perf + ranked perf: {perf + ranked_perf}')
        self.log("val_loss", perf, batch_size=batch_size, prog_bar=True, on_epoch=True)
        

    def test_step(self, batch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")
    
    def _evaluate_batch(self, batch, label: str):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, aux_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds, logits = self(bmg, V_d, X_d)
        weights = torch.ones_like(weights)

        
        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        for m in self.metrics:
            m.update(preds, targets)
            self.log(f"{label}/{m.alias}", m, batch_size=batch_size)
            # print(f"{label}/{m.alias}")
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        bmg, V_d, X_d, *_ = batch
        preds, logits = self(bmg, V_d, X_d)
        return {"preds": preds.cpu().numpy(), "logits": logits.cpu().numpy()}
