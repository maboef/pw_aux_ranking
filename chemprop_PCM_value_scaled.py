import os
import sys
import cupy
import mlflow
import joblib
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path

from src.auxiliary_chemprop_PCM import train_auxiliary_chemprop_PCM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rdkit.Chem import rdFingerprintGenerator


def add_protein_descriptors_and_pad(data, prot_descriptor_path):
    prot_descriptor = pd.read_pickle(prot_descriptor_path)
    # prot_descriptor['protein_descriptor'] = prot_descriptor['protein_descriptor'].apply(list)
    data = pd.merge(left=data, right=prot_descriptor, left_on='accession', right_on='target_id', how='left')
    missing = data[data['protein_descriptor'].isna()]
    print(f'{len(missing)} datapoints without protein descriptor')
    print(f"proteins without descriptor added: {(missing['accession'].unique())}")
    data = data[~data['protein_descriptor'].isna()].reset_index()
    return data

def create_apply_scaler(prot, model_path):
    if os.path.exists(model_path + '/scaler'):
        print('protein descriptor scaler found, loading...')
        scaler = joblib.load(model_path + '/scaler')
    else:
        print('no protein descriptor scaler associated yet')
        scaler = StandardScaler()
        scaler = scaler.fit(np.array(prot.to_list()))
        joblib.dump(scaler, model_path + '/scaler')
    scaled = scaler.transform(np.array(prot.to_list()))
    return scaled

def value_scaler(train, model_path):
    scaler_lookup = train.groupby('accession')['value'].agg(['mean', 'std']).reset_index()
    scaler_lookup['std'] = scaler_lookup['std'].replace(0, 1).fillna(1)
    scaler_path = os.path.join(model_path, 'target_scaler.csv')
    scaler_lookup.to_csv(scaler_path, index=False)
    return scaler_lookup, scaler_path

def run_model(
    data_path,
    protein_descriptor : str = 'One_Hot_Encoded',
    ranking : bool = True,
    extension: bool = True,
    seed  : int = 101,
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
    model_path = f'/home/boefma/auxiliary_ranking/models/value_scaled_{extension}/'
    Path(model_path).mkdir(parents=True, exist_ok=True)
    data_path = data_path + extension + '.csv'
    
    prot_descriptor_path = f'/home/boefma/auxiliary_ranking/data/protein_data/{protein_descriptor}.pkl'
    data = pd.read_csv(data_path)
    data['value'] = data['pchembl_value_Mean']
    data['subset'] = data['Subset']
    data['fixed_relation'] = data['relation']
    data = add_protein_descriptors_and_pad(data, prot_descriptor_path)
    prot = data['protein_descriptor']
    scaled = create_apply_scaler(prot, model_path)
    data['protein_descriptor'] = list(scaled)

    train = data[data['subset'] == 'train']
    train = train.reset_index()
    train_size = len(train)
    valid = data[data['subset'] == 'valid']
    valid = valid.reset_index()
    valid_size = len(valid)
    test = data[data['subset'] == 'test']
    test = test.reset_index()
    test_size = len(test)
    print(f'test {test_size}, train {train_size}, val {valid_size}')

    # value scaling
    scaler_lookup, scaler_path = value_scaler(train, model_path)
    train = train.merge(scaler_lookup, on='accession', how='left')
    train['value'] = (train['value'] - train['mean']) / train['std']
    valid = valid.merge(scaler_lookup, on='accession', how='left')
    valid['value'] = (valid['value'] - valid['mean']) / valid['std']
    print(len(train['rank_split'].unique()))
    if len(train['rank_split'].unique()) <5:
        categorical_dist = 0
    elif len(train['rank_split'].unique()) > 5:
        categorical_dist = 1
    else:
        print('number of unique values in rank split other than 3 or 6, check')
    with mlflow.start_run() as run:
        params = {
            'hidden_size': 512, 
            'depth': 3,
            'dropout': 0.1,
            'ffn_num_layers': 3,
            'activation': 'LeakyReLU',
            'aggregation': 'mean',
            'max_lr': 0.002,
            'epochs': epochs,
            'bias': False,
            'batch_size': 512,
            'extension': extension,
            'seed': seed,
            'rank_dist': 0.3,
            'categorical_dist': categorical_dist}
        model_path = model_path + run.data.tags.get("mlflow.runName")
        mlflow.log_params(params)
        trainer, model = train_auxiliary_chemprop_PCM(train, valid, model_path, **params)
        mlflow.end_run()
        return trainer, model, test, valid, train, model_path

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("PCM_models_saifudeen")
    for seed in range(5):
        seed = seed+1
        trainer, model, test_, valid, train, model_path = run_model(data_path='/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_', protein_descriptor='CMF_Zscales', extension='cluster_split_base', seed=seed) # saifudeen_ext_double_rank, saifudeen_ext_6_class)

    '''
    trainer, model, test_, valid, train = run_model(data_path='/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_', protein_descriptor='CMF_Zscales', extension='cluster_split_saifudeen_ext_double_rank', seed=1001)
    trainer, model, test_, valid, train = run_model(data_path='/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_', protein_descriptor='CMF_Zscales', extension='cluster_split_saifudeen_ext_double_rank', seed=10011)
    trainer, model, test_, valid, train = run_model(data_path='/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_', protein_descriptor='CMF_Zscales', extension='cluster_split_saifudeen_ext_double_rank', seed=20002)
    trainer, model, test_, valid, train = run_model(data_path='/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_', protein_descriptor='CMF_Zscales', extension='cluster_split_saifudeen_ext_double_rank', seed=30001)
    trainer, model, test_, valid, train = run_model(data_path='/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_', protein_descriptor='CMF_Zscales', extension='cluster_split_saifudeen_ext_double_rank', seed=10, epochs=2000)
    '''