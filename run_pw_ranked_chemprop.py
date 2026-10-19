import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.pw_ranked_chemprop_PCM import train_auxiliary_chemprop
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.auxiliary_ranking.rank_loss import Level, Cond, CrossRule, RankRuleConfig

from rdkit.Chem import rdFingerprintGenerator

'''
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
'''

cfg = RankRuleConfig(
    levels=(
        Level(target=0, margin=1.0),
        Level(target=1, margin=10.0),
    ),
    
    cross_rules=(
        CrossRule(winner=(Cond(0, ">", 7.0), Cond(1, "isnan")),
                  loser=(Cond(1, "<", 50),)),                  
        CrossRule(winner=(Cond(1, ">", 50), Cond(0, "isnan")),
                  loser=(Cond(0, "<", 5.5),))
    ),
    cross_rules_respect_veto=False
)


def value_scaler(train, model_path):
    scaler_path = (model_path / 'target_scaler.csv')
    if scaler_path.exists():
        scaler_lookup = pd.read_csv(scaler_path)
    else:
        scaler_lookup = train.groupby('accession')['value'].agg(['mean', 'std']).reset_index()
        scaler_lookup['std'] = scaler_lookup['std'].replace(0, 1).fillna(1)
        scaler_lookup.to_csv(scaler_path, index=False)
    return scaler_lookup, scaler_path

def run_model(
    dataset_dir,
    current_dir,
    # protein_descriptor : str = 'One_Hot_Encoded',
    ranking : bool = True,
    extension: bool = True,
    seed : int = 101,
    epochs : int = 500):
    
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
    data_path = data_path.with_suffix('.parquet')
    
    # prot_descriptor_path = dataset_dir / f'{protein_descriptor}_protein_descriptors.pkl'
    data = pd.read_parquet(data_path)
    data['value'] = data['pchembl_value_Mean']
    data['subset'] = data['Subset']
    data['fixed_relation'] = '='
    data['accession'] = data['target_id']
    # data['fixed_relation'] = data['relation']

    '''
    data = add_protein_descriptors_and_pad(data, prot_descriptor_path)
    prot = data['protein_descriptor']
    scaled = create_apply_scaler(prot, model_path)
    '''

    train = data[data['subset'] == 'train']
    train = train.reset_index(drop=True)
    train_size = len(train)
    valid = data[data['subset'] == 'valid']
    valid = valid[valid['fixed_relation'] == '='] # ensure none of the censored data is used for validation
    valid = valid[valid['pchembl_value_Mean'].notna()]
    valid = valid.reset_index(drop=True)
    valid_size = len(valid)
    test = data[data['subset'] == 'test']
    test = test[test['pchembl_value_Mean'].notna()]
    test = test.reset_index(drop=True)
    test_size = len(test)
    print(f'test {test_size}, train {train_size}, val {valid_size}')

    # value scaling
    scaler_lookup, scaler_path = value_scaler(train, model_path)
    train = train.merge(scaler_lookup, on='accession', how='left')
    train['value'] = (train['value'] - train['mean']) / train['std']
    valid = valid.merge(scaler_lookup, on='accession', how='left')
    valid['value'] = (valid['value'] - valid['mean']) / valid['std']
    print(train)
    params = {
        'hidden_size': 512, 
        'depth': 3,
        'dropout': 0.2,
        'ffn_num_layers': 3,
        'activation': 'LeakyReLU',
        'aggregation': 'mean',
        'max_lr': 0.002,
        'epochs': epochs,
        'bias': False,
        'batch_size': 512,
        'extension': extension,
        'seed': seed,
        'ranking_rules': cfg}
    model_path = (model_path / str(seed))
    trainer, model = train_auxiliary_chemprop(train, valid, model_path, **params)

if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent

    for seed in range(1):
        seed = seed+1
        run_model(dataset_dir=Path('/home/boefma/oncode_accelerator/avoidome/hERG_modelling/datasets/'), current_dir=current_dir, extension='herg_extended_test', seed=seed)
 