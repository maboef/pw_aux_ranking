
import io
import torch
from chemprop import models
from torch.nn import functional as F

import logging

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
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        bmg, V_d, X_d, *_ = batch
        preds, logits = self(bmg, V_d, X_d)
        return {"preds": preds.cpu().numpy(), "logits": logits.cpu().numpy()}
