import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.pw_ranked_chemprop_PCM import train_auxiliary_chemprop_PCM
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rdkit.Chem import rdFingerprintGenerator

def add_protein_descriptors_and_pad(data, prot_descriptor_path):
    prot_descriptor = pd.read_pickle(prot_descriptor_path)
    data = pd.merge(left=data, right=prot_descriptor, on='accession', how='left')
    missing = data[data['protein_descriptor'].isna()]
    print(f'{len(missing)} datapoints without protein descriptor')
    print(f"proteins without descriptor added: {(missing['accession'].unique())}")
    data = data[~data['protein_descriptor'].isna()].reset_index()
    return data

def create_apply_scaler(prot, model_path):
    if (model_path / 'scaler').exists():
        print('protein descriptor scaler found, loading...')
        scaler = joblib.load(model_path / 'scaler')
    else:
        print('no protein descriptor scaler associated yet')
        scaler = StandardScaler()
        scaler = scaler.fit(np.array(prot.to_list()))
        joblib.dump(scaler, model_path / 'scaler')
    scaled = scaler.transform(np.array(prot.to_list()))
    return scaled

def value_scaler(train, model_path):
    scaler_lookup = train.groupby('accession')['value'].agg(['mean', 'std']).reset_index()
    scaler_lookup['std'] = scaler_lookup['std'].replace(0, 1).fillna(1)
    scaler_path = (model_path / 'target_scaler.csv')
    scaler_lookup.to_csv(scaler_path, index=False)
    return scaler_lookup, scaler_path

def run_model(
    dataset_dir,
    current_dir,
    protein_descriptor : str = 'One_Hot_Encoded',
    ranking : bool = True,
    extension: bool = True,
    seed : int = 101,
    epochs : int = 100):
    
    """
    Run a model on a dataset.
    
    Parameters
    ----------
    model : str
        Name of the model to run.
    dataset : str
        Name of the dataset to use.
    split : str
        Name of the split to use.
    mode : str
        Name of the mode to use: ST or MT.
    params : str
        Name of the parameters to use.
    param_path : str
        Path to the parameters to use.
    imputation : str
        Name of the imputation method to use.
    """
    # If param_path is given, set params to HyperOpt istaed of the name of the method
    # Create paths
    model_path = (current_dir / 'models'/ extension)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    data_path = dataset_dir / extension
    data_path = data_path.with_suffix('.csv')
    
    prot_descriptor_path = dataset_dir / f'{protein_descriptor}_protein_descriptors.pkl'
    data = pd.read_csv(data_path)
    data['value'] = data['pchembl_value_Mean']
    data['subset'] = data['Subset']
    data['fixed_relation'] = data['relation']
    
    data = add_protein_descriptors_and_pad(data, prot_descriptor_path)
    prot = data['protein_descriptor']
    scaled = create_apply_scaler(prot, model_path)
    data['protein_descriptor'] = list(scaled)

    train = data[data['subset'] == 'train']
    train = train.reset_index(drop=True)
    train_size = len(train)
    valid = data[data['subset'] == 'valid']
    valid = valid[valid['fixed_relation'] == '='] # ensure none of the censored data is used for
    valid = valid.reset_index(drop=True)
    valid_size = len(valid)
    test = data[data['subset'] == 'test']
    test = test.reset_index(drop=True)
    test_size = len(test)
    print(f'test {test_size}, train {train_size}, val {valid_size}')

    # value scaling
    scaler_lookup, scaler_path = value_scaler(train, model_path)
    train = train.merge(scaler_lookup, on='accession', how='left')
    train['value'] = (train['value'] - train['mean']) / train['std']
    valid = valid.merge(scaler_lookup, on='accession', how='left')
    valid['value'] = (valid['value'] - valid['mean']) / valid['std']
    print(f"unique rank split values: {len(train['rank_split'].unique())}")
    if 'rank_split' not in train:
        train['rank_split'] = np.nan
    if len(train['rank_split'].unique()) <5:
        categorical_dist = 0
    elif len(train['rank_split'].unique()) > 5:
        categorical_dist = 1
    else:
        print('number of unique values in rank split other than 3 or 6, check')
    params = {
        'hidden_size': 512, 
        'depth': 3,
        'dropout': 0.1,
        'ffn_num_layers': 3,
        'activation': 'LeakyReLU',
        'aggregation': 'mean',
        'max_lr': 0.002,
        'epochs': epochs,
        'bias': True,
        'batch_size': 512,
        'extension': extension,
        'seed': seed,
        'rank_margin_1': 2, # margin for pchembl_value ranking
        'rank_margin_2' : 30, # margin for pct inhibition ranking
        'categorical_dist': categorical_dist}
    model_path = (model_path / str(seed))
    trainer, model = train_auxiliary_chemprop_PCM(train, valid, model_path, **params)

if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent
    for seed in range(10):
        seed = seed+1
        run_model(dataset_dir=current_dir / 'data/datasets/', current_dir=current_dir, protein_descriptor='Z-scales', extension='cluster_split_ext_rank_cont_set', seed=seed) 
