import os
import tqdm
import rdkit
import joblib
import pathlib
import numpy as np
import pandas as pd
import scipy
import torch

from sklearn import metrics
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from lightning import pytorch as pl
from sklearn.preprocessing import StandardScaler
from chemprop import data, featurizers, models, utils


def compute_fps(data):
    """Compute Morgan Fingerprints from SMILES."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fps = pd.DataFrame(np.array([mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in tqdm.tqdm(data.SMILES, desc='Computing Morgan Fingerprints from SMILES')]), index=data.index)
    return fps


def add_protein_descriptors_and_pad(data, prot_descriptor_path):
    prot_descriptor = pd.read_pickle(prot_descriptor_path)
    # prot_descriptor['protein_descriptor'] = prot_descriptor['protein_descriptor'].apply(list)
    print(prot_descriptor['protein_descriptor'].apply(type))
    data = pd.merge(left=data, right=prot_descriptor, left_on='accession', right_on='target_id', how='left')
    missing = data[data['protein_descriptor'].isna()]
    print(f'{len(missing)} datapoints without protein descriptor')
    print(f"proteins without descriptor added: {(missing['accession'].unique())}")
    data = data[~data['protein_descriptor'].isna()].reset_index()
    return data

def unscale_predictions(predictions, scaler_csv_path, global_mean=0, global_std=1):
    scaler_lookup = pd.read_csv(scaler_csv_path)
    output_df = pd.merge(predictions, scaler_lookup, on='accession', how='left')
    output_df['mean'] = output_df['mean'].fillna(global_mean)
    output_df['std'] = output_df['std'].fillna(global_std)
    output_df['unscaled_prediction'] = (output_df['prediction_0'] * output_df['std']) + output_df['mean']
    output_df = output_df.drop(columns=['mean', 'std'])
    return output_df


def predict_chemprop_PCM(dataset, model_path: str):
    smis = dataset.loc[:, 'SMILES'].values
    prots = dataset.loc[:, 'protein_descriptor'].values
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
    datapoints = [data.MoleculeDatapoint(mol=mol, x_d=prot) for mol, prot in zip(mols, prots)]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(datapoints, featurizer)
    test_loader = data.build_dataloader(test_dset, batch_size=500, shuffle=False)

    try:
        model = models.model.MPNN.load_from_checkpoint(model_path)
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator="gpu",
                devices=1
                )
            test_preds = trainer.predict(model, test_loader)
            test_preds = torch.vstack(test_preds)
            dataset['prediction'] = np.array(test_preds)
            return dataset
    except:
        model = torch.load(model_path, weights_only=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator="gpu",
                devices=1
            )
            test_preds = trainer.predict(model, test_loader)
            test_preds = torch.vstack(test_preds)
            dataset['prediction'] = np.array(test_preds[:, 0])
            dataset['logit_1'] = np.array(test_preds[:, 1:2])
            dataset['logit_2'] = np.array(test_preds[:, 2:3])
        return dataset


def generate_preds(model_path, scaler_path, data, value_scaler):
    preds, prot = run_model_inference_PCM('CP', 'PCM', model_path, scaler_path,
                                               data=data, protein_descriptor='CMF_Zscales')
    preds = unscale_predictions(preds, value_scaler)
    return preds


def get_performance(true, pred):
    rmse = metrics.root_mean_squared_error(true, pred)
    r2 = metrics.r2_score(true, pred)
    spearmanr = scipy.stats.spearmanr(true, pred)
    return rmse, r2, spearmanr