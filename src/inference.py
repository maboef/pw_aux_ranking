import os
import tqdm
import rdkit
import joblib
import pathlib
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile

from sklearn import metrics
import chemprop
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler



def performance(selected_set):
    mean_r2 = metrics.r2_score(selected_set['value'], selected_set['prediction'])
    r2_list = []
    rmse_list = []
    len_list = []
    for target in selected_set['accession'].unique():
        target_set = selected_set[selected_set['accession'] == target]
        true = target_set['value']
        pred = target_set['prediction']
        # r2 = metrics.r2_score(true, pred)
        if len(target_set) > 10:
            r2 = metrics.r2_score(true, pred)
            mse = metrics.root_mean_squared_error(true, pred)
            length = len(target_set)
        else:
            continue
        if r2 is not np.nan or float('nan'):
            print(f'target {target} - R2 score {r2} - rmse {mse} - length {length}')
            r2_list.append(r2)
            rmse_list.append(mse)
            len_list.append(length)
        else:
            print(f'target {target} score {r2}')
    print(mean_r2)
    print(np.median(r2_list))
    print(np.mean(r2_list))
    print(f'mean rmse: {np.mean(rmse_list)}')
    return performance_dict