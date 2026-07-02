
import torch

from chemprop.nn.hparams import HasHParams
from chemprop.nn.utils import Activation
import logging

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
