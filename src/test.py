import os
import math
import tqdm
import rdkit
import joblib
import pathlib
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from sklearn import metrics

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler

def compute_fps(data):
    """Compute Morgan Fingerprints from SMILES."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fps = pd.DataFrame(np.array([mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in tqdm.tqdm(data.SMILES, desc='Computing Morgan Fingerprints from SMILES')]), index=data.index)
    return fps

def mkdirs(path : str):
    """Create a directory if it does not exist."""
    if not os.path.exists(path) : os.makedirs(path)

def save_temp_df(df):
    tmp = NamedTemporaryFile(delete=False, suffix='.csv')
    name = pathlib.Path(tmp.name).name
    cwd = pathlib.Path.cwd()
    tmp_dir = cwd / 'tmp/'
    mkdirs(tmp_dir)
    path = tmp_dir / name
    df.to_csv(path, index=False)
    return str(path)

def censored_errors(preds, true, lt_mask, gt_mask):
    preds = np.where((preds < true) & lt_mask, true, preds)
    preds = np.where((preds > true) & gt_mask, true, preds)
    MAE = metrics.mean_absolute_error(true, preds)
    MSE = metrics.mean_squared_error(true, preds)
    return MAE, MSE

def get_preds(data, model_path, model):
    if model == 'CP':
        from .chemprop_PCM import predict_chemprop_PCM
        preds = predict_chemprop_PCM(data, model_path)
    else:
        from .rf_PCM import predict_RF_PCM
        from .pyboost_PCM import predict_PB_PCM
        from .xgboost_PCM import predict_XGB_PCM
    if model == 'PB':
        preds = predict_PB_PCM(data, model_path)
    if model == 'RF':
        preds = predict_RF_PCM(data, model_path)
    if model == 'XGB':
        preds = predict_XGB_PCM(data, model_path)
    return preds

def run_model_inference_PCM(model : str, 
                            dataset : str = 'PCM', 
                            split : str = 'luukkonen',
                            model_path : str = None,
                            data = None,
                            cens_data = None):
    preds = get_preds(data, model_path, model)
    preds.to_csv(model_path + '/preds_exact.csv', index=False)
    selected_set = preds[~preds['value'].isna()]
    mean_r2 = metrics.r2_score(selected_set['value'], selected_set['prediction'])
    r2_list = []
    for target in selected_set['accession'].unique():
        target_set = selected_set[selected_set['accession'] == target]
        true = target_set['value']
        pred = target_set['prediction']
        r2_list.append(metrics.r2_score(true, pred))
    r2_list = np.array(r2_list)
    r2_list = r2_list[~np.isnan(r2_list)]

    if cens_data is not None:
        preds = get_preds(cens_data, model_path, model)
        preds.to_csv(model_path + '/preds_censored.csv', index=False)
        selected_set = preds[~preds['value'].isna()]
        lt_mask = np.expand_dims(np.array(cens_data['relation'].str.contains('<').values), axis=1)
        gt_mask = np.expand_dims(np.array(cens_data['relation'].str.contains('>').values), axis=1)
        selected_set['lt'] = lt_mask
        selected_set['gt'] = gt_mask
        mae_list = []
        mse_list = []
        for target in selected_set['accession'].unique():
            target_set = selected_set[selected_set['accession'] == target]
            lt_mask = target_set['lt']
            gt_mask = target_set['gt']
            true = target_set['value']
            pred = target_set['prediction']
            mse, mae = censored_errors(pred, true, lt_mask, gt_mask) 
            mse_list.append(mse)
            mae_list.append(mae)
        mae, mse = censored_errors(selected_set['value'], selected_set['prediction'], selected_set['lt'], selected_set['gt'])
        return mean_r2, np.median(r2_list), np.mean(r2_list), mae, mse, np.median(mse_list), np.mean(mse_list), np.median(mae_list), np.mean(mae_list)
    return mean_r2, np.median(r2_list), np.mean(r2_list)