import pandas as pd
from pathlib import Path

from lightning import pytorch as pl

import torch
import numpy as np
from chemprop import nn, utils
from chemprop.data import split_data_by_indices
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from src.pl_checkpointer import TorchModelSaver
from src.auxiliary_ranking.aux_rank_reimplementation import CustomMPNN, MultiObjectiveFFN, MSEPlusPairwiseRankingLoss
from src.auxiliary_ranking.aux_rank_utils import Datum, TrainingBatch, CustomMoleculeDataset, custom_collate_batch, MoleculeDatapoint, custom_build_dataloader

from pytorch_lightning.loggers import TensorBoardLogger
from lightning import pytorch as pl

dataset = pd.read_csv('data/full_split.csv')

regression_dataset = dataset
regression_dataset['fixed_relation'] = '='
# regression_dataset.loc[regression_dataset['mean_pct_inhib'] > 20, 'class'] = 1
# regression_dataset.loc[regression_dataset['mean_pct_inhib'] < 15, 'class'] = 0
# regression_dataset['rank_split'] = regression_dataset['class']
regression_dataset['rank_split'] = regression_dataset['overlap_bin']
regression_train = regression_dataset[regression_dataset['subset'].isin(['train'])]
regression_validation = regression_dataset[regression_dataset['subset'].isin(['validation'])]
regression_validation = regression_validation[~regression_validation['pchembl_value_Mean'].isna()]
regression_test = regression_dataset[regression_dataset['subset'].isin(['test'])]


