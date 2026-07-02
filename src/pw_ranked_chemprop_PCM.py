import os
import sys
import mlflow
import numpy as np
import pandas as pd

from chemprop import nn, utils
from chemprop.data import split_data_by_indices
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from lightning.pytorch import seed_everything
from lightning import pytorch as pl
from .pl_checkpointer import TorchModelSaver
from .auxiliary_ranking.custom_mpnn import CustomMPNN
from .auxiliary_ranking.rank_ffn import MultiObjectiveFFN
from .auxiliary_ranking.rank_loss import MSEPlusPairwiseRankingLoss
from .auxiliary_ranking.utils import Datum, TrainingBatch, CustomMoleculeDataset, custom_collate_batch, MoleculeDatapoint, custom_build_dataloader


import torch
torch.set_float32_matmul_precision("high")


def train_auxiliary_chemprop_PCM(train, valid, model_path: str,  **kwargs):
    """
    Train a chemprop model for the whole dataset (v2.1.2 style).
    """
    if kwargs:
        params = kwargs
    else:
        with open(model_path) as d: params = json.load(d)

    seed_everything(params['seed'], workers=True)
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
    
    if 'rank_split' and 'scaled_pchembl_value' and 'scaled_percent_inhibition' in dataset:
        aux_mask = dataset[['rank_split', 'scaled_pchembl_value', 'scaled_percent_inhibition']].to_numpy()
        print(f'aux_mask unique: {aux_mask.shape}')
    elif 'rank_split' in dataset:
        dataset['scaled_pchembl_value'] = np.nan
        dataset['scaled_percent_inhibition'] = np.nan
        aux_mask = dataset[['rank_split', 'scaled_pchembl_value', 'scaled_percent_inhibition']].to_numpy()
        print(f'aux_mask unique: {aux_mask.shape}')
    else:
        print('no rank split column found')
        aux_mask = np.expand_dims(np.ones(len(ys)), axis=1)

    
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
    datapoints = [MoleculeDatapoint(mol=mol, y=[float(y)], x_d=prot, lt_mask=lt, gt_mask=gt, aux_mask=aux) 
                  for mol, y, prot, lt, gt, aux in zip(mols, ys, prots, lt_mask, gt_mask, aux_mask)] # using the pre-existing weight value for storing and use of target indices as it's anyway passed along all through chemprop to the loss function
    train_data, val_data, test_data = split_data_by_indices([datapoints], *split_indices)
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    
    train_dset = CustomMoleculeDataset(train_data[0][0], featurizer)
    val_dset = CustomMoleculeDataset(val_data[0][0], featurizer)
    print(f'length train_dset {len(train_dset)}')
    if 'rank_split' in dataset:
        train_loader = custom_build_dataloader(train_dset, batch_size=batch_size, shuffle=True, class_balance=False, drop_last=True, seed=params['seed'])
        val_loader = custom_build_dataloader(val_dset, batch_size=batch_size, shuffle=False, class_balance=False)
        print('class balance set to true')
    else:
        train_loader = custom_build_dataloader(train_dset, batch_size=batch_size, shuffle=True, class_balance=False, drop_last=True, seed=params['seed'])
    
    mp = nn.BondMessagePassing(d_h=params['hidden_size'], # hidden size (v1: hidden_size=300)
                               depth=params['depth'], # message-passing steps (v1: depth=3)
                               dropout=params['dropout'],      # (v1 default 0.0)
                               activation=params['activation'],
                               bias=params['bias'], # v1 used bias=False in W_i/W_h
                               undirected=False)

    ffn_input_dim = mp.output_dim + prots[0].shape[0]
    ffn = MultiObjectiveFFN(input_dim=ffn_input_dim,
                            hidden_dim=params['hidden_size'],
                            n_tasks=1,
                            n_layers=params['ffn_num_layers'],
                            dropout=params['dropout'],
                            activation=params['activation'],
                            criterion=MSEPlusPairwiseRankingLoss(rank_dist=params['rank_dist'],
                                                                 categorical_dist=params['categorical_dist']))

    if params['aggregation'] == 'mean':
        agg = nn.MeanAggregation()
    if params['aggregation'] == 'sum':
        agg = nn.SumAggregation()
    metric_list = [nn.metrics.RMSE()]
    chemprop_model = CustomMPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list, max_lr=params['max_lr'])
    trainer = pl.Trainer(logger=True, enable_checkpointing=False, max_epochs=params['epochs'], accelerator='gpu', 
                         devices=[3], deterministic=True, callbacks=[TorchModelSaver(
        dirpath=model_path,
        filename="best-{epoch}-val_rmse",
        monitor="val/rmse",
        mode='min')])
    trainer.fit(chemprop_model, train_loader, val_loader)
    return trainer, chemprop_model


def predict_chemprop_PCM(dataset, model_path: str):
    model = CustomMPNN.load_from_checkpoint(model_path)
    smis = dataset.loc[:, 'SMILES'].values
    prots = dataset.loc[:, 'protein_descriptor'].values
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
    datapoints = [data.MoleculeDatapoint(mol=mol, x_d=prot) for mol, prot in zip(mols, prots)]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(datapoints, featurizer)
    test_loader = data.build_dataloader(test_dset, batch_size=50, shuffle=False)
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1)
    test_preds = trainer.predict(model, test_loader)