import random
import torch
from torch.amp import autocast
from chemprop.nn import metrics
from chemprop import data, featurizers, models, nn

from chemprop.nn.utils import Activation


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
        nll_loss = torch.sum(-targets * log_pred, dim=1)
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

class MultiObjectiveFFN(torch.nn.Module):
    """
    Wrapper around RegressionFFN that adds multi-objective loss capability.
    Instead of wrapping the base_ffn, we create our own FFN structure.
    """
    n_targets = 1
    def __init__(self, input_dim, hidden_dim, n_tasks, n_layers, dropout, activation, 
                 bce_weight=1.0, criterion=None):
        super().__init__()
        
        self.bce_weight = bce_weight
        
        # Use the provided criterion or default to MSEPlusPairwiseRankingLoss
        if criterion is None:
            self.criterion = MSEPlusPairwiseRankingLoss()
        else:
            self.criterion = criterion
        
        # Build FFN layers manually (matching RegressionFFN structure)
        layers = []
        
        # First layer
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        
        # Output layer
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_dim, n_tasks))
        
        self.ffn = torch.nn.Sequential(*layers)
        
        # Identity transform (no scaling)
        self.output_transform = torch.nn.Identity()
        self.arc_classifier = torch.nn.Linear(n_tasks, 2)
        
        # Store hyperparameters
        self.hparams = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'n_tasks': n_tasks,
            'n_layers': n_layers,
            'dropout': dropout,
            'activation': activation
        }

    def forward(self, x, targets):
        """
        Forward pass - returns predictions.
        x: FFN input (batch, input_dim)
        Returns: predictions (batch, n_tasks)
        """
        # Get predictions from FFN
        preds = self.ffn(x)
        logits_arc =  self.arc_classifier(preds)
        logits_mat, targets_mat = self.compute_rank_logits(logits_arc, targets)
        return self.output_transform(preds), logits_mat, logits_arc, targets_mat

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

    def calc_loss(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None, rank_split=None):
        """
        Calculate combined loss for regression + pairwise ranking.
        
        preds: regression predictions (batch, tasks)
        targets: ground truth values (batch, tasks)
        mask: validity mask (batch, tasks)
        weights: sample weights (batch, tasks)
        lt_mask: less-than mask for bounded loss (batch, tasks)
        gt_mask: greater-than mask for bounded loss (batch, tasks)
        rank_split: ranking group indicator (batch,) - 0 for negatives, 1 for positives
        """
        # Generate binary predictions (for classification head)
        bin_preds = torch.sigmoid(preds)
        # Calculate combined loss using your custom criterion
        # print(f'rank_split {rank_split.unique(return_counts=True)}')
        loss = self.criterion._calc_unreduced_loss(
            preds=preds,
            bin_preds=bin_preds,
            targets=targets,
            mask=mask,
            weights=weights,
            lt_mask=lt_mask,
            gt_mask=gt_mask,
            rank_split=rank_split
        )
        return loss


class MSEPlusPairwiseRankingLoss(metrics.ChempropMetric):
    def __init__(self, num_pairs: int = 50):
        super().__init__()
        self.num_pairs = num_pairs
        self.mse = nn.BoundedMSE()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.alias = 'Combined_regression_rank_loss'

    def _calc_unreduced_loss(self, preds, bin_preds, targets, mask=None,
                             weights=None, lt_mask=None, gt_mask=None,
                             rank_split=None):
        """
        preds:       (batch, tasks) - regression predictions
        bin_preds:   (batch, tasks) - binary predictions (sigmoid of preds)
        targets:     (batch, tasks) - ground truth
        mask:        (batch, tasks) or None
        lt_mask:     (batch, tasks) - less than mask
        gt_mask:     (batch, tasks) - greater than mask
        rank_split:  (batch,) tensor of 0 or 1 marking which group the sample belongs to
        """

        # ---------- MSE LOSS ----------
        
        mask = targets.isfinite()
        mse_loss = self.mse(preds, targets, mask, weights, lt_mask, gt_mask)
        
        # ---------- PAIRWISE RANKING LOSS ----------
        if rank_split is None:
            # If no rank_split provided, return just MSE loss
            return mse_loss
        # print(f'targets: {targets}')
        # print(f'preds: {preds}')
        # print(f'bin_preds: {bin_preds}')
        num_tasks = preds.size(1)

        # Indices of the two groups
        pos_idx = torch.where(rank_split == 1)[0]
        neg_idx = torch.where(rank_split == 0)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            # No mixed pairs possible
            return mse_loss
        
        pair_losses = []
        
        for t in range(num_tasks):
            # Valid samples for this task under masking
            if mask is not None:
                valid = torch.where(mask[:, t] > 0)[0]
                pos_valid = [i for i in pos_idx.tolist() if i in valid.tolist()]
                neg_valid = [i for i in neg_idx.tolist() if i in valid.tolist()]
            else:
                pos_valid = pos_idx.tolist()
                neg_valid = neg_idx.tolist()

            if len(pos_valid) == 0 or len(neg_valid) == 0:
                continue

            # Sample pairs: each negative matched with random positive
            for _ in range(self.num_pairs):
                i = random.choice(neg_valid)  # negative example
                j = random.choice(pos_valid)  # positive example
                y_i = preds[i, t]
                y_j = preds[j, t]
                t_i = targets[i, t]
                t_j = targets[j, t]
                # Label is 1 if target_i > target_j
                label = (t_i > t_j).float()

                # Logit is score difference
                logit = y_i - y_j

                loss_ij = self.bce(logit.unsqueeze(0), label.unsqueeze(0))
                pair_losses.append(loss_ij)
        if len(pair_losses) > 0:
            pair_loss = torch.stack(pair_losses).mean()
        else:
            pair_loss = torch.tensor(0., device=preds.device)
        # print(f'mse_loss: {mse_loss}')
        # print(f'pair_loss: {pair_loss}')
        return mse_loss + pair_loss


class CustomMPNN(models.MPNN):
    """Custom MPNN that handles multi-objective loss with ranking."""
    
    def _compute_loss(self, preds, targets, logits_mat, targets_mat, mask, weights, lt_mask, gt_mask, aux_mask, phase="train"):
        """Override loss computation to include rank_split for pairwise ranking."""
        # Get predictions
        
        # Extract rank_split from the batch if available
        # If predictor has custom loss calculation (MultiObjectiveFFN)
        if hasattr(self.predictor, 'calc_loss'):
            loss = self.predictor.calc_loss(
                preds=preds,
                targets=targets,
                mask=mask,
                weights=weights,
                lt_mask=lt_mask,
                gt_mask=gt_mask,
                rank_split=aux_mask
            )
        else:
            # Fallback to standard criterion
            loss = self.predictor.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
        
        return loss

    '''
    def training_step(self, batch, batch_idx):
        bmg, x_d, targets, mask, weights, lt_mask, gt_mask, aux_mask = batch
        loss, preds = self._compute_loss(bmg, x_d, targets, mask, weights, lt_mask, gt_mask, aux_mask, "train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bmg, x_d, targets, mask, weights, lt_mask, gt_mask, aux_mask = batch
        loss, preds = self._compute_loss(bmg, x_d, targets, mask, weights, lt_mask, gt_mask, "val")
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        for metric in self.metrics:
            val = metric(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"val/{metric.alias}", val, prog_bar=True)
        
        return loss
    '''
    
    def training_step(self, batch, batch_idx):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, aux_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        with autocast():
            Z = self.fingerprint(bmg, V_d, X_d)
            preds, logits_mat, logits_arc, targets_mat = self.predictor.forward(Z, targets)
            loss = self._compute_loss(preds, targets, logits_mat, targets_mat, mask, weights, lt_mask, gt_mask, aux_mask, phase="train")
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, aux_mask = batch

        mask = targets.isfinite()
        mask = mask.float() if mask is not None else None
        # targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds, logits_mat, targets_mat = self.predictor.forward(Z, targets)
        loss = self._compute_loss(preds, targets, mask, weights, lt_mask, gt_mask, aux_mask, phase="valid")
        self.log("val_loss", loss, batch_size=batch_size, prog_bar=True)

    def test_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, aux_mask = batch

        mask = targets.isfinite()
        mask = mask.bool() if mask is not None else None
        
        Z = self.fingerprint(bmg, V_d, X_d)
        preds, logits_mat, targets_mat = self.predictor.forward(Z, targets)
        loss = self._compute_loss(preds, targets, mask, weights, lt_mask, gt_mask, aux_mask, phase="test")
        
        # Log metrics
        self.log("test_loss", loss)
        for metric in self.metrics:
            val = metric(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"test/{metric.alias}", val)

        return loss

    def _evaluate_batch(self, batch, label: str) -> None:
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, aux_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        Z = self.fingerprint(bmg, V_d, X_d)
        preds, logits_mat, targets_mat = self.predictor.forward(Z, targets)
        weights = torch.ones_like(weights)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=batch_size)