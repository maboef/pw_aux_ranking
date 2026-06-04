import os
import tqdm
import joblib
import time
import operator

import cupy as cp
import numpy as np
import pandas as pd

from src.utils import mkdirs, compute_fps
from src.optim_custom_boosting import OptimGradientBoosting
# from src.custom_boosting import GradientBoosting
from src.custom_losses import MSELoss

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from py_boost.callbacks.callback import Callback
from py_boost.gpu.losses.metrics import RMSEMetric
from py_boost.multioutput.sketching import RandomProjectionSketch

from scipy.stats import rv_continuous, rv_discrete
import mlflow

class pyboost_grad_monitor(Callback):
    """
    class for logging of gradient development during PyBoost training
    """
    def __init__(self, verbose):
        self.verbose = verbose
        
    def after_iteration(self, build_info):
        grad = build_info['data']['train']['grad']
        num_iter = build_info['num_iter']
        
        if (((num_iter) % self.verbose) == 0):
            grad_sum = grad.sum()
            grad_max = grad.max()
            grad_mean = grad.mean()
            print(f'grad mean: {grad_mean}')
            if mlflow.active_run() is not None:
                mlflow.log_metric('grad sum', grad.sum())
                mlflow.log_metric('grad max', grad.max())
                mlflow.log_metric('grad mean', grad.mean())


def relations_to_operators_python(relations, false_relations: bool = False):
    """
    transforms a pandas dataframe containing relations into python operators
    investigate whether this can be transformed into numpy/cupy version 
    
    args:
        relations: pandas dataframe containing relations in string ( '<', '=', '=' etc.)\
        false_relations: bool : specifying whether to set all relations to eq, for sanity testing etc. 
    return:
        operators: pandas dataframe containing python operators
    """
    eq = operator.eq
    ne = operator.ne
    lt = operator.lt
    le = operator.le
    gt = operator.gt
    ge = operator.ge
    if false_relations:
        operators = relations.replace({'=' : eq, '!=' : eq, '<' : eq, '<=' : eq, '>' : eq, '>=' : eq}) 
        
    else:
        operators = relations.replace({'=' : eq, '!=' : ne, '<' : lt, '<=' : le, '>' : gt, '>=' : ge})

    return operators

class censored_MSEWithNaNLoss(MSELoss):
    """
    censored MSE regression loss based on SurvLoss: A New Survival Loss Function for Neural Networks to
    Process Censored Data - by Rahat et al. (2024)
    """
    def __init__(self, ):
        
        self.feats_cols = None

    def __call__(self, y_true, y_pred, relation):
        y_pred = cp.concatenate(y_pred)
        grad, hess = self.get_grad_hess(y_true, y_pred, relation)
        return grad, hess
    
    def get_relation_mask_operator(self, y_true, y_pred, operator):
        """
        Create mask for censored data - mask where y_pred is in censored area will return True - non-censored area will return False
        """
        relation_mask = np.array([op(a, b) for a, b, op in zip(y_pred, y_true, operator)])
        return relation_mask
    
    def get_grad_hess(self, y_true, y_pred, relation):
        """
        Args: 
            y_true: array of ground truth values (both exact and censored data) (n_samples,)
            y_pred: array of predicted values (n_samples,)
            relation: array of length (n_samples,) containing relation
        """
        relation_mask = ~self.get_relation_mask_operator(y_true.get(), y_pred.get(), relation)
        # apply features mask
        relation_mask = cp.array(relation_mask)
        grad = y_pred - np.where(relation_mask, y_true, 0)
        grad = grad
        hess = relation_mask.astype(np.float32)
        grad *= hess
        
        # we will ignore not only NaNs but also columns that are used as features !!!
        if self.feats_cols is not None:
            hess[:, self.feats_cols] = 0
            grad *= hess
        hess = np.expand_dims(hess, axis=1)
        grad = np.expand_dims(grad, axis=1)
        return grad, hess
        
    def base_score(self, y_true):
        """This method defines how to initialize the ensemble
        
        Args:
            y_true: cp.ndarray of target values
            
        Returns:

        """
        print(f'base score: {cp.array([cp.nanmean(y_true, axis=0)], dtype=cp.float32)}')
        return cp.array([cp.nanmean(y_true, axis=0)], dtype=cp.float32)

class RMSEWithNaNMetric(RMSEMetric):
    """
    This is custom MSE Loss that accepts NaN values and ignores features
    """
    def __init__(self):
        """
        
        Args:
            target_cols: list of int, indices of columns that could be both features and targets
            
        Returns:

        """
        print('initiating RMSEWithNaNMetric')

    
    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        
        Args:
            y_true: cp.ndarray of target values
            y_pred: cp.ndarray of predicted values
            sample_weight: cp.ndarray of sample weights or None
            
        Returns:

        """       
        
        mask = ~cp.isnan(y_true)
        
        err = (cp.where(mask, y_true, 0) - y_pred) ** 2
        return err[mask].mean() ** .5

# Function to create and train a model

def PB_hyperparams():
    """
    Returns a dictionary of hyperparameters for the PyBoost model
    """

    return {
        "lr":  [0.005, 0.01, 0.05],
        "max_depth": [8, 10, 12],
        "lambda_l2": [5, 10, 20, 50],
        "colsample": [1.0],  # [0.6, 0.7, 0.8, 0.9, 1.0],  sampling methods break when relation is used
        "subsample": [1.0],  #  [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_gain_to_split": [0, 1, 2],
        "gd_steps": [1], #, 2, 5, 10],
        "quantization": ['Quantile', 'Uniform', 'Uniquant']
    }

def train_censored_PB_PCM_opt(
    train,
    valid,
    model_path : str,
    seed : int = 2022,
    n_iter : int = 100,
    verbose : int = 100
    ):

    mkdirs(model_path)

    fps = compute_fps(valid).astype(float).values
    X_val = np.concatenate((fps, np.array(valid['protein_descriptor'].tolist())), axis=1)
    y_val = valid['value']
    relation_val = valid['relation']
    operator_val = relations_to_operators_python(relation_val, false_relations=False)

    fps = compute_fps(train).astype(float).values
    X_train = np.concatenate((fps, np.array(train['protein_descriptor'].tolist())), axis=1)
    y_train = train['value']
    relation_train = train['relation']
    operator_train = relations_to_operators_python(relation_train, false_relations=False)
    print(operator_val.value_counts())
    print(operator_train.value_counts())
    eval_sets = [{'X': X_val, 'y': y_val}]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    param_list = list(ParameterSampler(PB_hyperparams(), n_iter=n_iter, random_state=seed))

    loss = censored_MSEWithNaNLoss()
    metric = RMSEWithNaNMetric()
    # sketch = RandomProjectionSketch(1)
    logger = pyboost_grad_monitor(verbose)

    best_score = -np.inf
    best_params = None
    best_model = None

    for j, params in tqdm.tqdm(enumerate(param_list), total=len(param_list)):

        print('Training model', j, 'with params', params)

        model = OptimGradientBoosting(
            loss, logger, metric=metric, seed=seed, verbose=verbose,
            ntrees=10000, es=250, **params
            )
        model.fit(X_train, y_train, relation=operator_train, eval_sets=eval_sets)
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
            try:
                r2_list.append(r2_score(true, pred))
                print(f'success: {target} - length true {len(true)}, length pred {len(pred)}')
            except:
                print(f'{target} - length true {len(true)}, length pred {len(pred)}')
                print(true)
                print(pred)
                print('no R2 defined - skipping')
        score = np.median(r2_list)
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
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


def predict_PB_PCM(data, model_path : str, preds_path : str = None):

    """
    Predicts the targets for a dataset using PyBoost
    
    Parameters
    ----------
    data_path : str
        Path to the data
    model_path : str
        Path to the models
    preds_path : str
        Path to save the predictions
    
    Returns
    -------
    preds : pd.DataFrame
        Predictions
    """

    preds = data.copy()

    fps = compute_fps(data).astype(float).values
    X_val = np.concatenate((fps, np.array(data['protein_descriptor'].tolist())), axis=1)
    
    model = joblib.load(model_path)
    predictions = model.predict(X_val)
    preds['prediction'] = predictions

    if preds_path is not None:
        os.makedirs(os.path.dirname(preds_path), exist_ok=True)
        preds.to_csv(preds_path, index=False)

    return preds