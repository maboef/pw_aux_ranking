import os
import tqdm
import rdkit
import pathlib
import numpy as np
import pandas as pd
import joblib
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
    # print(prot_descriptor['protein_descriptor'].apply(type))
    data = pd.merge(left=data, right=prot_descriptor, left_on='accession', right_on='target_id', how='left')
    missing = data[data['protein_descriptor'].isna()]
    # print(f'{len(missing)} datapoints without protein descriptor')
    # print(f"proteins without descriptor added: {(missing['accession'].unique())}")
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


def predict_chemprop_PCM(dataset, model_paths: list):
    """
    Args:
        dataset: DataFrame with 'SMILES' column
        model_paths: List of paths to the 5 CV fold models
    Returns:
        dataset: DataFrame with added columns for mean prediction, std, and individual fold predictions
    """
    smis = dataset.loc[:, 'SMILES'].values
    prots = dataset.loc[:, 'protein_descriptor'].values
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
    datapoints = [data.MoleculeDatapoint(mol=mol, x_d=prot) for mol, prot in zip(mols, prots)]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(datapoints, featurizer)
    test_loader = data.build_dataloader(test_dset, batch_size=100, shuffle=False)
    
    all_predictions = []

    for i, model_path in enumerate(model_paths):
        print(f"Running inference on model {i+1}/{len(model_paths)}")
        
        model = torch.load(model_path, weights_only=False)
        
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator="gpu",
                devices=1
            )
            
            raw_preds = trainer.predict(model, test_loader)
            # print(f'raw preds: {raw_preds}')
            all_values = [list(batch.values())[0] for batch in raw_preds]
            logits = [list(batch.values())[1] for batch in raw_preds]
            # 2. Flatten and convert to a single NumPy array
            test_preds = np.concatenate([v.cpu().numpy() if torch.is_tensor(v) else v for v in all_values], axis=0)
            test_logits = np.concatenate([v.cpu().numpy() if torch.is_tensor(v) else v for v in logits], axis=0)
            preds_df = pd.DataFrame(test_preds).add_prefix('prediction_')
            logits_df = pd.DataFrame(test_logits).add_prefix('logits_')

            dataset = pd.concat([dataset.reset_index(drop=True), preds_df], axis=1)
            dataset = pd.concat([dataset, logits_df], axis=1)
    return dataset


def run_model_inference_PCM(model : str, 
    dataset : str = 'PCM', 
    model_paths : list = None,
    scaler_path : str = None,
    data = None,
    protein_descriptor : str = 'kin_AFembeddings_pca256'):
    
    prot_descriptor_path = f'data/protein_data/{protein_descriptor}.pkl'
    data = add_protein_descriptors_and_pad(data, prot_descriptor_path)
    prot = data['protein_descriptor']

    if scaler_path is not None:
        # print(os.path.split(scaler_path)[0] + '/scaler')
        if os.path.exists(os.path.split(scaler_path)[0] + '/scaler'):
            # print('protein descriptor scaler found, loading...')
            scaler = joblib.load(os.path.split(scaler_path)[0]+ '/scaler')
        else:
            # print('no protein descriptor scaler associated yet')
            scaler = StandardScaler()
            scaler = scaler.fit(np.array(prot.to_list()))
            joblib.dump(scaler, os.path.splitext(scaler_path)[0] + '.scaler')
        scaled = scaler.transform(np.array(prot.to_list()))
        data['protein_descriptor'] = list(scaled)
    
    else:
        data['protein_descriptor'] = list(np.array(prot.to_list()))
        
    if model == 'CP':
        preds = predict_chemprop_PCM(data, model_paths)
        preds = preds.drop(labels='protein_descriptor', axis=1)
    return preds, prot


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


if __name__ == '__main__':
    random_test = pd.read_csv('/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_no_saifudeen_ext.csv')
    random_test = random_test[random_test['Subset'] == 'test']
    random_test = random_test[random_test['relation'] == '=']
    random_test['value'] = random_test['pchembl_value_Mean']
    
    
    cluster_test = pd.read_csv('/home/boefma/auxiliary_ranking/data/PCM_luukkonnen_cluster_split_no_saifudeen_ext.csv')
    cluster_test = cluster_test[cluster_test['Subset'] == 'test']
    cluster_test = cluster_test[cluster_test['relation'] == '=']
    cluster_test['value'] = cluster_test['pchembl_value_Mean']
    
    saifudeen_test = pd.read_parquet('/zfsdata/data/boefma/kinase_data/saifudeen_data_dose_curves.parquet')
    saifudeen_test['accession'] = saifudeen_test['Uniprot ID'] + '_WT'
    saifudeen_test['value'] = saifudeen_test['pEC50']
    saifudeen_test = saifudeen_test[saifudeen_test['pEC50'].notna()]
    
    saifudeen_train = pd.read_parquet('/zfsdata/data/boefma/kinase_data/saifudeen_data_pct_inhibition.parquet')

    base_path = '/home/boefma/auxiliary_ranking/models/'
    
    split = 'value_scaled_cluster_split_saifudeen_ext_6_class/'
    models = ['adaptable-eel-165/best-35-val_rmse.pt',
              'whimsical-stork-147/best-62-val_rmse.pt',
              'adaptable-frog-643/best-65-val_rmse.pt',
              'painted-cow-342/best-65-val_rmse.pt',
              'serious-dolphin-492/best-73-val_rmse.pt']

    for model in models:
        model_path = os.path.join(base_path, split, model)
        prot_scaler_path = os.path.join(base_path, split)
        y_scaler_path = os.path.join(base_path, split, 'target_scaler.csv')

        random_preds = generate_preds([model_path], prot_scaler_path, random_test, y_scaler_path)
        print(f'RMSE, R2, Spearman R: {get_performance(random_preds.value, random_preds.unscaled_prediction)}')
        
        cluster_preds = generate_preds([model_path], prot_scaler_path, cluster_test, y_scaler_path)
        print(f'RMSE, R2, Spearman R: {get_performance(cluster_preds.value, cluster_preds.unscaled_prediction)}')
        
        # preds_1 = preds_1[preds_1['prediction_0'] != preds_1['unscaled_prediction']]
        # luukkonen_preds_1 = generate_preds(model_paths_1, scaler_path_ext_cont, luukkonen_data, value_scaler_ext_cont)
        # luukkonen_preds_cluster_1 = generate_preds(model_paths_1, scaler_path_ext_cont, luukkonen_data_cluster, value_scaler_ext_cont)