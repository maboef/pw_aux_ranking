import os
import tqdm
import joblib

import mlflow
import cloudpickle
import cupy as cp
import numpy as np
import pandas as pd
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import AllChem
from src.utils import mkdirs, compute_fps
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rv_continuous, rv_discrete

# Class to handle missing values in a PyBoost multi-task model ################

'''
class MSEWithNanLoss(MSELoss):
    """
    This is custom MSE Loss that accepts NaN values and ignores features
    """
    
    def __init__(self, ):
        self.feats_cols = None
    def get_grad_hess(self, y_true, y_pred):
        """
        Args:
            y_true: cp.ndarray of target values
            y_pred: cp.ndarray of predicted values
        Returns:

        """
        mask = ~cp.isnan(y_true)
        # apply features mask
        grad = y_pred - cp.where(mask, y_true, 0)
        hess = mask.astype(cp.float32)
        grad *= hess
        # we will ignore not only NaNs but also columns that are used as features !!!
        if self.feats_cols is not None:
            hess[:, self.feats_cols] = 0
            grad *= hess
        return grad, hess

    def base_score(self, y_true):
        """This method defines how to initialize the ensemble
        Args:
            y_true: cp.ndarray of target values
        Returns:
        """
        return cp.nanmean(y_true, axis=0)

        
class RMSEWithNaNMetric(RMSEMetric):
    """
    This is custom MSE Loss that accepts NaN values and ignores features
    """
    def __init__(self, target_cols):
        """
        Args:
            target_cols: list of int, indices of columns that could be both features and targets
            
        Returns:

        """
        self.target_cols = target_cols

    
    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        
        Args:
            y_true: cp.ndarray of target values
            y_pred: cp.ndarray of predicted values
            sample_weight: cp.nndarray of sample weights or None
            
        Returns:

        """
        y_true = y_true[:, self.target_cols]
        y_pred = y_pred[:, self.target_cols]         
        
        mask = ~cp.isnan(y_true)
        
        err = (cp.where(mask, y_true, 0) - y_pred) ** 2
        return err[mask].mean() ** .5
'''

def XGB_hyperparams():
    """
    Returns a dictionary of hyperparameters for XGBoost
    """
    return {"max_depth": [3, 6, 12, 16],
                "learning_rate": [0.01,0.02,0.03,0.04,0.05],
                "min_child_weight":[1, 2, 4, 6, 8, 16,24],
                "n_estimators": [100, 200, 400, 800, 4000],
                "colsample_bytree": [0.1, 0.2, 0.3],
                "subsample": [0.7, 0.8,0.9,1],
                "scale_pos_weight": [10,25,30,50]
            }

def train_XGB_QSAR_reg_opt(train, valid, model_path, seed=2022, n_iter=100):
    # 1. Initialize Modern Generator
    fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def get_fp_array(df):
        mols = [Chem.MolFromSmiles(s) for s in df['SMILES']]
        fps = [fp_gen.GetFingerprintAsNumPy(m) for m in mols if m is not None]
        return np.stack(fps).astype(np.float32)

    print("Generating Fingerprints...")
    X_train = get_fp_array(train)
    y_train = train['pchembl_value_Mean'].values.astype(np.float32)
    
    X_val = get_fp_array(valid)
    y_val = valid['pchembl_value_Mean'].values.astype(np.float32)

    eval_set = [(X_val, y_val)]
    param_list = list(ParameterSampler(XGB_hyperparams(), n_iter=n_iter, random_state=seed))

    best_score = -np.inf
    best_model = None

    print("Starting Hyperparameter Optimization...")
    for params in tqdm.tqdm(param_list):
        model = xgb.XGBRegressor(
            **params, 
            random_state=seed, 
            tree_method='hist', 
            device="cuda"
        )
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False # Keep the progress bar clean
        )

        # Validation score (assuming RMSE, so we want the lowest)
        score = model.score(X_val, y_val) 
        print(f'model score: {score}')
        print(f'best score: {best_score}')
        if score > best_score:
            best_score = score
            best_model = model

    # Save logic...
    return best_model

def train_XGB_QSAR_class_opt(train, valid, model_path, seed=2022, n_iter=100):
    # 1. Initialize Modern Generator
    fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def get_fp_array(df):
        mols = [Chem.MolFromSmiles(s) for s in df['SMILES']]
        fps = [fp_gen.GetFingerprintAsNumPy(m) for m in mols if m is not None]
        return np.stack(fps).astype(np.float32)

    print("Generating Fingerprints...")
    X_train = get_fp_array(train)
    y_train = train['class'].values.astype(np.float32)
    
    X_val = get_fp_array(valid)
    y_val = valid['class'].values.astype(np.float32)

    eval_set = [(X_val, y_val)]
    param_list = list(ParameterSampler(XGB_hyperparams(), n_iter=n_iter, random_state=seed))

    best_score = -np.inf
    best_model = None

    print("Starting Hyperparameter Optimization...")
    for params in tqdm.tqdm(param_list):
        model = xgb.XGBClassifier(
            **params, 
            random_state=seed, 
            tree_method='hist', 
            device="cuda"
        )
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False # Keep the progress bar clean
        )

        # Validation score (assuming RMSE, so we want the lowest)
        score = model.score(X_val, y_val) 
        print(f'model score: {score}')
        print(f'best score: {best_score}')
        if score > best_score:
            best_score = score
            best_model = model

    # Save logic...
    return best_model



"""

def train_XGB_QSAR_opt(
    train,
    valid,
    model_path : str,
    # logging_params : dict,
    seed : int = 2022,
    n_iter : int = 10,
    ):

    mkdirs(model_path)
    train['split_index'] = -1
    valid['split_index'] = 0
    
    train['morgan_fp'] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) 
                        for m in [Chem.MolFromSmiles(s) for s in train['SMILES']]]
    valid['morgan_fp'] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) 
                        for m in [Chem.MolFromSmiles(s) for s in valid['SMILES']]]
    X_val = np.array(valid['morgan_fp'].tolist())
    y_val = np.array(valid['pchembl_value_Mean'])
    X = np.array(train['morgan_fp'].tolist())
    y = np.array(train['pchembl_value_Mean'])
    eval_sets = [{'X': cp.array(X_val), 'y': cp.array(y_val)}]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    param_list = list(ParameterSampler(XGB_hyperparams(), n_iter=n_iter, random_state=seed))

    best_score = -np.inf
    best_params = None
    best_model = None
    for j, params in tqdm.tqdm(enumerate(param_list), total=len(param_list)):
        model = xgb.XGBRegressor(**params, random_state = seed, tree_method='hist', device = "cuda:0") # # **params,
        model.fit(cp.array(X), cp.array(y), eval_set=eval_sets)
        try :
            y_pred = model.predict(X_val)
            y_pred = pd.DataFrame(y_pred)
            true = y_val
            score = matthews_corrcoef(true, y_pred)
            if score > best_score:
                print('Best median MCC score:', matthews_corrcoef(true, y_pred))
                joblib.dump(best_model, model_path + '/model.joblib')
                best_score=score
        except:
            print('Warning: Failed to fit the model or predict on validation set') # Free memory
            del model
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            continue    
            # score = r2_score(y_val, y_pred)
            # Free memory
            del model
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
    return model

def predict_XGB_PCM(data, model_path : str):
    preds = data.copy()
    fps = compute_fps(data).astype(float).values
    X_val = np.concatenate((fps, np.array(data['protein_descriptor'].tolist())), axis=1)
    model = joblib.load(model_path + '/model.joblib')
    predictions = model.predict(X_val)
    preds['prediction'] = predictions
    return preds
"""