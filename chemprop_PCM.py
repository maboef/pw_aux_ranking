import os
import sys
import cupy
import joblib
from torch import nn
import numpy as np
import pandas as pd

from src.auxiliary_chemprop_PCM import train_auxiliary_chemprop_PCM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rdkit.Chem import rdFingerprintGenerator

def compute_fps(data):
    """Compute Morgan Fingerprints from SMILES."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fps = pd.DataFrame(np.array([mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in tqdm.tqdm(data.SMILES, desc='Computing Morgan Fingerprints from SMILES')]), index=data.index)

    # fps = pd.DataFrame(np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=2048) for smiles in tqdm.tqdm(data.SMILES, desc='Computing Morgan Fingerprints from SMILES')]), index=data.index)

    return fps
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

def run_model(
    data_path,
    protein_descriptor : str = 'One_Hot_Encoded',
    ranking : bool = True,
    extension: bool = True):
    
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
    model_path = f'/home/boefma/auxiliary_ranking/models/{extension}/'
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
    params = {
        'hidden_size' : 256, 
        'depth' : 3,
        'dropout' : 0.01,
        'ffn_num_layers' : 3,
        'activation' : 'LeakyReLU',
        'aggregation' : 'mean',
        'max_lr' : 0.0025,
        'epochs' : 200,
        'bias' : False,
        'batch_size': 512}

    trainer, model = train_auxiliary_chemprop_PCM(train, valid, model_path, **params)
    return trainer, model, test, valid, train

trainer, model, test_, valid, train = run_model(data_path='/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_', protein_descriptor='CMF_Zscales', extension='no_saifudeen_ext') # no_ext)