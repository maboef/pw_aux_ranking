import os
import sys
import mlflow
import numpy as np
import pandas as pd
from chemprop import data, featurizers, models, nn
from chemprop.nn import metrics
from chemprop.models import multi
from chemprop import utils
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from .pl_checkpointer import TorchModelSaver

def train_censored_chemprop_PCM(train, valid, model_path: str,  **kwargs):
    """
    Train a chemprop model for the whole dataset (v2.1.2 style).
    """
    if kwargs:
        print(kwargs)
        params = kwargs
    else:
        with open(model_path) as d: params = json.load(d)
    
    batch_size = params['batch_size']
    dataset = pd.concat([train, valid], ignore_index=True)
    
    train_idx = list(range(0, len(train)))
    valid_idx = list(range(len(train), len(train)+len(valid)))
    split_indices = tuple([[train_idx], [valid_idx]])
    
    smis = dataset.loc[:, 'SMILES'].values
    ys = dataset.loc[:, 'value'].values
    prots = dataset.loc[:, 'protein_descriptor'].values
    lt_mask = np.expand_dims(np.array(dataset['fixed_relation'].str.contains('<').values), axis=1)
    gt_mask = np.expand_dims(np.array(dataset['fixed_relation'].str.contains('>').values), axis=1)
    target_indices = np.array(pd.factorize(dataset['accession'])[0])
    
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
    datapoints = [data.MoleculeDatapoint(mol=mol, y=[float(y)], x_d=prot, lt_mask=lt, gt_mask=gt) for mol, y, prot, lt, gt in zip(mols, ys, prots, lt_mask, gt_mask)]
    train_data, val_data, test_data = data.split_data_by_indices(datapoints, *split_indices)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    train_loader = data.build_dataloader(train_dset, batch_size=batch_size)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, shuffle=False)

    mp = nn.BondMessagePassing(d_h=params['hidden_size'], # hidden size (v1: hidden_size=300)
                               depth=params['depth'], # message-passing steps (v1: depth=3)
                               dropout=params['dropout'],      # (v1 default 0.0)
                               activation=params['activation'],
                               bias=params['bias'], # v1 used bias=False in W_i/W_h
                               undirected=False)
    
    ffn_input_dim = mp.output_dim + prots[0].shape[0]
    ffn = nn.RegressionFFN(hidden_dim=params['hidden_size'], input_dim = ffn_input_dim, n_tasks = 1, n_layers=params['ffn_num_layers'], dropout=params['dropout'], activation=params['activation'], criterion=metrics.BoundedRMSE())
    
    if params['aggregation'] == 'mean':
        agg = nn.MeanAggregation()
    if params['aggregation'] == 'sum':
        agg = nn.SumAggregation()
    metric_list = [metrics.RMSE()]
    chemprop_model = models.MPNN(mp, agg, ffn, batch_norm=params['batch_norm'], metrics=metric_list, max_lr=params['max_lr'])
    # checkpoint_callback = ModelCheckpoint(model_path, 'best-{epoch}-{val_loss:.2f}', save_top_k=1, monitor="val/rmse", mode='min')
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, max_epochs=params['epochs'], accelerator='gpu', devices=1, callbacks=[TorchModelSaver(
        dirpath=model_path,
        filename="best-{epoch}",
        monitor="val/rmse",
        mode='min')])
    trainer.fit(chemprop_model, train_loader, val_loader)
    # results_val = trainer.test(chemprop_model, val_loader, "best")
    return trainer # , results_val, results_test, test_loader, chemprop_model

def predict_chemprop_PCM(dataset, model_path: str):
    model = models.model.MPNN.load_from_checkpoint(model_path)
    smis = dataset.loc[:, 'SMILES'].values
    prots = dataset.loc[:, 'protein_descriptor'].values
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
    datapoints = [data.MoleculeDatapoint(mol=mol, x_d=prot) for mol, prot in zip(mols, prots)]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(datapoints, featurizer)
    test_loader = data.build_dataloader(test_dset, batch_size=50, shuffle=False)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1
        )
    test_preds = trainer.predict(model, test_loader)