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
from lightning import pytorch as pl
from sklearn.preprocessing import StandardScaler
from chemprop import data, featurizers, models, utils


def add_protein_descriptors_and_pad(data, prot_descriptor_path):
    prot_descriptor = pd.read_pickle(prot_descriptor_path)
    # prot_descriptor['accession'] = prot_descriptor['accession'] + '_WT'
    data = pd.merge(left=data, right=prot_descriptor, on ='accession', how='left')
    missing = data[data['protein_descriptor'].isna()]
    print(f'{len(missing)} datapoints without protein descriptor')
    print(f"proteins without descriptor added: {(missing['accession'].unique())}")
    data = data[~data['protein_descriptor'].isna()].reset_index()
    return data

def unscale_predictions(predictions, scaler_csv_path, global_mean=0, global_std=1):
    scaler_lookup = pd.read_csv(scaler_csv_path)
    # scaler_lookup['accession'] = scaler_lookup['accession'] + '_WT'
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
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model = model.to("cuda")
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator="gpu",
                devices=[0]
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
    model_paths : list = None,
    scaler_path : str = None,
    data = None,
    protein_descriptor : str = 'Z-scales'):
    
    prot_descriptor_path = f'data/datasets/{protein_descriptor}_protein_descriptors.pkl'
    data = add_protein_descriptors_and_pad(data, prot_descriptor_path)
    prot = data['protein_descriptor']

    if scaler_path is not None:
        # print(os.path.split(scaler_path)[0] + '/scaler')
        if os.path.exists(os.path.split(scaler_path)[0] + '/scaler'):
            scaler = joblib.load(scaler_path + '/scaler')
        else:
            print('no protein descriptor scaler associated yet')
        scaled = scaler.transform(np.array(prot.to_list()))
        data['protein_descriptor'] = list(scaled)
    
    else:
        data['protein_descriptor'] = list(np.array(prot.to_list()))
        
    if model == 'CP':
        preds = predict_chemprop_PCM(data, model_paths)
        preds = preds.drop(labels='protein_descriptor', axis=1)
    return preds


def generate_preds(model_path, scaler_path, data, value_scaler):
    preds = run_model_inference_PCM('CP', model_path, scaler_path,
                                               data=data, protein_descriptor='Z-scales')
    preds = unscale_predictions(preds, value_scaler)
    return preds


def get_performance(true, pred):
    rmse = metrics.root_mean_squared_error(true, pred)
    r2 = metrics.r2_score(true, pred)
    spearmanr = scipy.stats.spearmanr(true, pred)
    return rmse, r2, spearmanr


def find_model_path(base_path, split, seed):
    """Find the single best-*-val_rmse.pt checkpoint in a seed's directory."""
    seed_dir = pathlib.Path(base_path) / split / str(seed)
    matches = sorted(seed_dir.glob('best-*-val_rmse.pt'))
    if not matches:
        raise FileNotFoundError(f'No model checkpoint found in {seed_dir}')
    if len(matches) > 1:
        raise ValueError(f'Multiple model checkpoints found in {seed_dir}: {matches}')
    return matches[0]


if __name__ == '__main__':
    current_dir = pathlib.Path(__file__).resolve().parent
    censored_test = pd.read_csv(current_dir / 'data/datasets/cluster_split_base_set.csv')
    censored_test = censored_test[censored_test['Subset'].isin(['test', 'valid'])]
    censored_test = censored_test[censored_test['fixed_relation'] != '=']
    censored_test['value'] = censored_test['pchembl_value_Mean']
    
    random_test = pd.read_csv(current_dir / 'data/datasets/random_split_base_set.csv')
    random_test = random_test[random_test['Subset'] == 'test']
    random_test = random_test[random_test['fixed_relation'] == '=']
    random_test['value'] = random_test['pchembl_value_Mean']
    
    
    cluster_test = pd.read_csv(current_dir / 'data/datasets/cluster_split_base_set.csv')
    cluster_test = cluster_test[cluster_test['Subset'] == 'test']
    cluster_test = cluster_test[cluster_test['relation'] == '=']
    cluster_test['value'] = cluster_test['pchembl_value_Mean']
    
    saifudeen_test = pd.read_csv(current_dir / 'data/datasets/saifudeen_pec50_data.csv')
    saifudeen_test['accession'] = saifudeen_test['Entry']
    saifudeen_test['value'] = saifudeen_test['pEC50']
    saifudeen_test = saifudeen_test[saifudeen_test['pEC50'].notna()]
    
    saifudeen_train = pd.read_csv(current_dir / 'data/datasets/saifudeen_percent_inhibition_data.csv')
    saifudeen_train['accession'] = saifudeen_train['Entry']
    saifudeen_train = saifudeen_train[saifudeen_train.percent_inhibition.notna()]
    
    base_path = '/home/boefma/publication_pw_aux_rank/pw_aux_rank/models/'
    
    '''
    
    split = 'cluster_split_base_set/'
    models = [# '1/best-51-val_rmse.pt',
              # '2/best-58-val_rmse.pt',
              '3/best-93-val_rmse.pt',
              # '4/best-91-val_rmse.pt',
              # '5/best-96-val_rmse.pt',
              '6/best-92-val_rmse.pt',
              # '7/best-69-val_rmse.pt',
              # '8/best-60-val_rmse.pt',
                '9/best-75-val_rmse.pt',
                '10/best-51-val_rmse.pt'
         ]
    '''
    
    
    split = 'cluster_split_base_rank_set/'
    models = [#'1/best-68-val_rmse.pt',
              # '2/best-93-val_rmse.pt',
              '3/best-74-val_rmse.pt',
              # '4/best-84-val_rmse.pt',
              # '5/best-83-val_rmse.pt',
              # '6/best-71-val_rmse.pt',
              # '7/best-79-val_rmse.pt',
              # '8/best-52-val_rmse.pt',
              # '9/best-75-val_rmse.pt',
              # z'10/best-74-val_rmse.pt'
             ]
    
    
    
    '''
    split = 'cluster_split_ext_rank_set/'
    models = [# '1/best-39-val_rmse.pt',
              # '2/best-77-val_rmse.pt',
              # '3/best-52-val_rmse.pt',
              # '4/best-66-val_rmse.pt',
              # '5/best-72-val_rmse.pt',
              # '6/best-60-val_rmse.pt',
              # '7/best-88-val_rmse.pt',
              # '8/best-98-val_rmse.pt',
              # '9/best-64-val_rmse.pt',
            '10/best-71-val_rmse.pt',
         ]
    
    '''
    '''
    split = 'cluster_split_ext_rank_cont_set/'
    models = [# '1/best-60-val_rmse.pt',
              # '2/best-61-val_rmse.pt',
              '3/best-67-val_rmse.pt',
              # '4/best-92-val_rmse.pt',
              # '5/best-45-val_rmse.pt',
              '6/best-87-val_rmse.pt',
              # '7/best-43-val_rmse.pt',
              # '8/best-72-val_rmse.pt',
              '9/best-80-val_rmse.pt',
              '10/best-50-val_rmse.pt',
             ]
    '''
    
    for model in models:
        model_path = os.path.join(base_path, split, model)
        prot_scaler_path = os.path.join(base_path, split)
        y_scaler_path = os.path.join(base_path, split, 'target_scaler.csv')
        model_dir = pathlib.Path(model_path).parent

        '''
        random_preds = generate_preds([model_path], prot_scaler_path, random_test, y_scaler_path)
        print(f'random split test - RMSE, R2, Spearman R: {get_performance(random_preds.value, random_preds.unscaled_prediction)}')
        random_preds.to_csv(f'{model_dir}/random_test_preds.csv', index=False)
        '''
        
        cluster_preds = generate_preds([model_path], prot_scaler_path, cluster_test, y_scaler_path)
        print(f'cluster split test - RMSE, R2, Spearman R: {get_performance(cluster_preds.value, cluster_preds.unscaled_prediction)}')
        cluster_preds.to_csv(f'{model_dir}/cluster_test_preds.csv', index=False)

        saifudeen_test_preds = generate_preds([model_path], prot_scaler_path, saifudeen_test, y_scaler_path)
        saifudeen_test_preds = saifudeen_test_preds[saifudeen_test_preds['prediction_0'] != saifudeen_test_preds['unscaled_prediction']]
        print(f'saifudeen test set - RMSE, R2, Spearman R: {get_performance(saifudeen_test_preds.value, saifudeen_test_preds.unscaled_prediction)}')
        saifudeen_test_preds.to_csv(f'{model_dir}/saifudeen_test_preds.csv', index=False)

        '''
        saifudeen_train_preds = generate_preds([model_path], prot_scaler_path, saifudeen_train, y_scaler_path)
        saifudeen_train_preds = saifudeen_train_preds[saifudeen_train_preds['prediction_0'] != saifudeen_train_preds['unscaled_prediction']]
        saifudeen_train_preds.to_csv(f'{model_dir}/saifudeen_train_preds.csv', index=False)

        censored_test_preds = generate_preds([model_path], prot_scaler_path, censored_test, y_scaler_path)
        censored_test_preds.to_csv(f'{model_dir}/censored_test_preds.csv', index=False)
        '''