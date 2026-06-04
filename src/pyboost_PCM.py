import os
import tqdm
import joblib
import mlflow
import cupy as cp
import numpy as np
import pandas as pd

from src.utils import mkdirs, compute_fps

from py_boost import GradientBoosting # basic GradientBoosting class
from py_boost.gpu.losses import * # utils for the custom loss
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rv_continuous, rv_discrete

# Class to handle missing values in a PyBoost multi-task model ################
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



def PB_hyperparams():
    """
    Returns a dictionary of hyperparameters for the PyBoost model
    """

    return {
        "lr":  [0.005, 0.01, 0.05],
        "max_depth": [12, 14, 16, 18],
        "lambda_l2": [5, 10, 20, 50],
        "colsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_gain_to_split": [0, 1, 2],
        "gd_steps": [1, 2, 5, 10],
        "quantization": ['Uniform', 'Uniquant'],
    }


def train_PB_PCM_opt(
    train,
    valid,
    model_path : str,
    seed : int = 2022,
    n_iter : int = 100,
    ):

    mkdirs(model_path)
    fps = compute_fps(valid).astype(float).values
    X_val = np.concatenate((fps, np.array(valid['protein_descriptor'].tolist())), axis=1)
    y_val = valid['value']
    
    fps = compute_fps(train).astype(float).values
    X_train = np.concatenate((fps, np.array(train['protein_descriptor'].tolist())), axis=1)
    y_train = train['value']
    eval_sets = [{'X': X_val, 'y': y_val}]
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    param_list = list(ParameterSampler(PB_hyperparams(), n_iter=n_iter, random_state=seed))

    best_score = -np.inf
    best_params = None
    best_model = None
    for j, params in tqdm.tqdm(enumerate(param_list), total=len(param_list)):
        print('Training model', j, 'with params', params)
        model = GradientBoosting(
            loss='mse', seed=seed, verbose=1000,
            ntrees=20000, es=250, **params
            )
        model.fit(X_train, y_train, eval_sets=eval_sets)
        try :
            y_pred = model.predict(X_val)
            y_pred = pd.DataFrame(y_pred)
        except:
            print('Warning: Failed to fit the model or predict on validation set')
            # Free memory
            del model
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            continue

        # Compute score : mean r2
        targets = list(valid['accession'].unique())

        
        # score = r2_score(y_val, y_pred)
        valid['preds'] = y_pred
        r2_list = []
        for target in targets:
            target_set = valid[valid['accession'] == target]
            true = target_set['value']
            pred = target_set['preds']
            r2_list.append(r2_score(true, pred))
        score = np.median(r2_list)
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
            mlflow.log_param("trial_number", j)
            mlflow.log_params(params)
            mlflow.log_metric('median R2 / target', score)
            joblib.dump(best_model, model_path + '/model.joblib')

        print('Score:', score)
        print(f'Best score: {best_score:.4f} with params {best_params}')

        # Free memory
        del model
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    joblib.dump(best_model, model_path + '/model.joblib')
    print('Best model saved to', f'{model_path}/model.joblib')
    print('Best score:', best_score)
    print('Best params:', best_params)


def predict_PB_PCM(data, model_path : str):
    preds = data.copy()
    fps = compute_fps(data).astype(float).values
    X_val = np.concatenate((fps, np.array(data['protein_descriptor'].tolist())), axis=1)
    model = joblib.load(model_path + '/model.joblib')
    predictions = model.predict(X_val)
    preds['prediction'] = predictions
    return preds

