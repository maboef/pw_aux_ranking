import os
import tqdm
import joblib

import cupy as cp
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from src.utils import mkdirs, compute_fps

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error, matthews_corrcoef
from scipy.stats import rv_continuous, rv_discrete

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

def RF_hyperparams():

    """
    Returns a dictionary of hyperparameters for Random Forests
    """
    
    return {
        'n_estimators': [100, 200, 400, 1000, 2000], # , 4000, 8000, 16000],
        'max_depth': [2, 4, 8, 12], #, 16, 24],
        'min_samples_split': [2, 6, 12, 24],
        'min_samples_leaf': [1, 2, 4, 8 , 12],
        'max_features': [1.0, 'sqrt', 'log2'],
    }

def train_RF_PCM_opt(
    train,
    valid,
    model_path : str,
    seed : int = 2022,
    n_iter: int = 100,
    ):

    mkdirs(model_path)
    n_jobs=8
    metric = root_mean_squared_error
    params = {
    "objective": "regression",  # or "binary", "multiclass", etc.
    "metric": "rmse"}

    fps = compute_fps(valid).astype(float).values
    X_val = np.concatenate((fps, np.array(valid['protein_descriptor'].tolist())), axis=1)
    y_val = valid['value']
    fps = compute_fps(train).astype(float).values
    X_train = np.concatenate((fps, np.array(train['protein_descriptor'].tolist())), axis=1)
    y_train = train['value']
    y_train = pd.DataFrame(y_train)
    y_train['split_index'] = -1

    y_val = pd.DataFrame(y_val)
    y_val['split_index'] = 0
    
    X = np.concatenate((X_train, X_val), axis=0)
    y = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    
    splits = PredefinedSplit(test_fold = y.split_index)
    model = RandomForestRegressor(random_state = seed, n_jobs=n_jobs)
    rdnSearch = RandomizedSearchCV(model, RF_hyperparams(), n_iter=n_iter, cv=splits, verbose=3, n_jobs=4, scoring=make_scorer(r2_score), random_state=seed)
    
    print('start search')
    rdnSearch.fit(X, y)
    joblib.dump(rdnSearch.best_estimator_, f'{model_path}/model.joblib')

    targets = list(valid['accession'].unique())
    # score = r2_score(y_val, y_pred)
    model = joblib.load(f'{model_path}/model.joblib')
    y_pred = model.predict(X_val)
    valid['preds'] = y_pred[:, 0]
    valid['probs'] = y_pred[:, 1]
    r2_list = []
    for target in targets:
        target_set = valid[valid['accession'] == target]
        true = target_set['value']
        pred = target_set['preds']
        r2_list.append(r2_score(true, pred))
    print(rdnSearch.best_params_)
    print(len(r2_list))
    print(np.mean(r2_list))
    score = np.median(r2_list)
    print('Best median R2 score:', score)
    # print(f'Best score: {score:.4f} with params {rdnSearch.best_params_}')

def predict_RF_PCM(data, model_path : str):
    preds = data.copy()
    fps = compute_fps(data).astype(float).values
    X_val = np.concatenate((fps, np.array(data['protein_descriptor'].tolist())), axis=1)
    model = joblib.load(model_path + '/model.joblib')
    predictions = model.predict(X_val)
    preds['prediction'] = predictions[:, 0]
    preds['probs'] = predictions[:, 1]
    return preds


def train_RF_QSAR_reg_opt(
    train,
    valid,
    model_path : str,
    seed : int = 2022,
    n_iter: int = 100,
    ):

    mkdirs(model_path)
    n_jobs=8
    metric = root_mean_squared_error
    train['split_index'] = -1
    valid['split_index'] = 0
    
    train['morgan_fp'] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) 
                        for m in [Chem.MolFromSmiles(s) for s in train['SMILES']]]
    valid['morgan_fp'] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) 
                        for m in [Chem.MolFromSmiles(s) for s in valid['SMILES']]]
    X_val = np.array(valid['morgan_fp'].tolist())
    y_val = np.array(valid['pchembl_value_Mean'])
    X = np.concat([np.array(train['morgan_fp'].tolist()), np.array(valid['morgan_fp'].tolist())], axis=0)
    y = np.concat([np.array(train['pchembl_value_Mean']), np.array(valid['pchembl_value_Mean'])], axis=0)
    split_index = np.concat([np.array(train['split_index']), np.array(valid['split_index'])], axis=0)
    splits = PredefinedSplit(test_fold = split_index)
    model = RandomForestRegressor(random_state = seed, n_jobs=n_jobs)
    rdnSearch = RandomizedSearchCV(model, RF_hyperparams(), n_iter=n_iter, cv=splits, verbose=3, n_jobs=4, scoring=make_scorer(r2_score), random_state=seed)
    
    print('start search')
    rdnSearch.fit(X, y)
    joblib.dump(rdnSearch.best_estimator_, f'{model_path}/model.joblib')

    targets = list(valid['accession'].unique())
    # score = r2_score(y_val, y_pred)
    model = joblib.load(f'{model_path}/model.joblib')
    y_pred = model.predict(X_val)
    true = y_val
    print('Best median R2 score:', r2_score(true, y_pred))

def train_RF_QSAR_class_opt(
    train,
    valid,
    model_path : str,
    seed : int = 2022,
    n_iter: int = 100,
    ):

    mkdirs(model_path)
    n_jobs=8
    train['split_index'] = -1
    valid['split_index'] = 0
    
    train['morgan_fp'] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) 
                        for m in [Chem.MolFromSmiles(s) for s in train['SMILES']]]
    valid['morgan_fp'] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) 
                        for m in [Chem.MolFromSmiles(s) for s in valid['SMILES']]]
    X_val = np.array(valid['morgan_fp'].tolist())
    y_val = np.array(valid['class'])
    X = np.concat([np.array(train['morgan_fp'].tolist()), np.array(valid['morgan_fp'].tolist())], axis=0)
    y = np.concat([np.array(train['class']), np.array(valid['class'])], axis=0)
    split_index = np.concat([np.array(train['split_index']), np.array(valid['split_index'])], axis=0)
    splits = PredefinedSplit(test_fold = split_index)
    model = RandomForestClassifier(random_state = seed, n_jobs=n_jobs)
    rdnSearch = RandomizedSearchCV(model, RF_hyperparams(), n_iter=n_iter, cv=splits, verbose=3, n_jobs=4, scoring=make_scorer(r2_score), random_state=seed)
    
    print('start search')
    rdnSearch.fit(X, y)
    joblib.dump(rdnSearch.best_estimator_, f'{model_path}/model.joblib')

    targets = list(valid['accession'].unique())
    # score = r2_score(y_val, y_pred)
    model = joblib.load(f'{model_path}/model.joblib')
    y_pred = model.predict(X_val)
    true = y_val
    print(np.isnan(true))
    print('Best median MCC score:', matthews_corrcoef(true, y_pred))
    return model