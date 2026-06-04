import os
import sys
import tqdm
import json
import pathlib

import numpy as np
import pandas as pd

from chemprop.args import TrainArgs, PredictArgs
from chemprop.train import cross_validate, run_training, make_predictions
from tempfile import NamedTemporaryFile

from .utils import mkdirs

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def save_temp_df(df):
    tmp = NamedTemporaryFile(delete=False, suffix='.csv')
    name = pathlib.Path(tmp.name).name
    cwd = pathlib.Path.cwd()
    tmp_dir = cwd / 'tmp/'
    mkdirs(tmp_dir)
    path = tmp_dir / name
    df.to_csv(path, index=False)
    return str(path)


def save_temp_npy(np_arr):
    tmp = NamedTemporaryFile(delete=False, suffix='.npy')
    name = pathlib.Path(tmp.name).name
    cwd = pathlib.Path.cwd()
    tmp_dir = cwd / 'tmp/'
    mkdirs(tmp_dir)
    path = tmp_dir / name
    np.save(path, np_arr, allow_pickle=True)
    return str(path)


def create_temp_path():
    tmp = NamedTemporaryFile(delete=False, suffix='.csv')
    name = pathlib.Path(tmp.name).name
    cwd = pathlib.Path.cwd()
    tmp_dir = cwd / 'tmp/'
    mkdirs(tmp_dir)
    path = tmp_dir / name
    return path


def train_chemprop_PCM(train, valid, test,
                       model_path: str, param_path: str = None, **kwargs):
    """
    Train a chemprop model for the whole dataset.
    Parameters
    ----------
    data_path : str
        Path to the training data.
    valid_path : str
        Path to the validation data.
    test_path : str
        Path to the test data.
    model_path : str
        Path to the directory where the models will be saved.
    param_path : str, optional
        Path to the json file containing the parameters for the model, by default None
    """

    os.makedirs(model_path, exist_ok=True)

    # Save temp files (Chemprop v1.7.1 requires file-based access)
    train_path = save_temp_df(train[['SMILES', 'value']])
    valid_path = save_temp_df(valid[['SMILES', 'value']])
    test_path = save_temp_df(test[['SMILES', 'value']])
    # feats = np.concatenate([np.vstack(train['protein_descriptor'].values), np.vstack(valid['protein_descriptor'].values), np.vstack(test['protein_descriptor'].values)], axis=0)
    # print(feats.shape[0])
    # print(len(train) + len(valid) + len(test))

    # features_path = save_temp_npy(feats)
    features_path = save_temp_npy(np.vstack(train['protein_descriptor'].values))
    features_valid_path = save_temp_npy(np.vstack(valid['protein_descriptor'].values))
    features_test_path = save_temp_npy(np.vstack(test['protein_descriptor'].values))
    # Base args

    cmd = '--data_path {} '.format(train_path)
    cmd += '--separate_val_path {} '.format(valid_path)
    cmd += '--separate_test_path {} '.format(test_path)

    cmd += '--dataset_type regression '
    cmd += '--smiles_columns SMILES '
    cmd += '--metric rmse '
    cmd += '--extra_metrics r2 '
    cmd += '--aggregation norm '
    cmd += '--quiet '
    cmd += '--gpu 1 '
    
    cmd += '--separate_val_features_path {} '.format(features_valid_path)
    cmd += '--separate_test_features_path {} '.format(features_test_path)
    cmd += '--features_path {} '.format(features_path)
    
    # Load from JSON param file if given
    if param_path :
        with open(param_path) as d: params = json.load(d)
        print(params)
        for k, v in params.items(): 
            cmd += f'--{k} {v} '

    # Add/override with kwargs    
    if kwargs:
        for k, v in kwargs.items(): cmd += f'--{k} {v} '
    
    cmd += '--save_dir {} '.format(model_path)
    
    # Parse args and train
    args = TrainArgs().parse_args(cmd.split())
    cross_validate(args=args, train_func=run_training)

    # Clean up temp files
    os.remove(train_path)
    os.remove(valid_path)
    os.remove(test_path)
    os.remove(features_path)
    os.remove(features_valid_path)
    os.remove(features_test_path)


def predict_chemprop_PCM(data, model_path: str):
    """
    Predict the activity of a dataset using a chemprop multi-task model.

    Parameters
    ----------
    data_path : str
        Path to the data.
    model_path : str
        Path to the directory where the models are saved.
    preds_path : str
        Path to the file where the predictions will be saved.
    Returns
    -------
    preds : pd.DataFrame
        Dataframe containing the predictions.
    """

    preds_path = create_temp_path()
    data_path = save_temp_df(data['SMILES'])
    features_path = save_temp_npy(np.vstack(data['protein_descriptor'].values))
    cmd = '--test_path {} '.format(data_path)
    cmd += '--features_path {} '.format(features_path)
    cmd += '--checkpoint_path {} '.format(model_path)
    cmd += '--preds_path {} '.format(preds_path)

    preds = make_predictions(args=PredictArgs().parse_args(cmd.split()))
    os.remove(data_path)
    os.remove(preds_path)
    os.remove(features_path)
    data['prediction'] = np.vstack(preds)
    return data

