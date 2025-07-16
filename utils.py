import joblib
import pandas as pd
import torch
import numpy as np
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
from torch.nn import Module as TorchModule


def apply_log_transform(df: pd.DataFrame, features_to_log: tuple):
    """
    Applies natural logarithm transformation (log1p) to selected features in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the features;
        features_to_log (tuple): Names of features to apply np.log1p() to.

    Returns:
        pd.DataFrame: A copy of the DataFrame with log-transformed features.
    """

    df = df.copy()

    for feature in features_to_log:
        df[feature] = np.log1p(df[feature])

    return df


def save_model(model, path: str):
    """
    Generic model saving.

    Parameters:
        model: Trained model (sklearn, XGBoost or PyTorch);
        path (str): Path to file to save the model.
    """

    if isinstance(model, BaseEstimator):
        # sklearn-model (including RandomForest)
        joblib.dump(model, path)
    elif isinstance(model, XGBRegressor):
        # XGBoost
        joblib.dump(model, path)
    elif isinstance(model, TorchModule):
        # PyTorch model
        torch.save(model.state_dict(), path)
    else:
        raise ValueError("Unsupported model type for saving")


def load_model(model_type: str, path: str, model_class=None):
    """
    Generic model loading.

    Parameters:
        model_type (str): 'random_forest', 'xgboost' or 'torch';
        path (str): Path to saved file;
        model_class: Model class (required for PyTorch).

    Returns:
        Loaded model.
    """

    if model_type == "random_forest":
        return joblib.load(path)
    elif model_type == "xgboost":
        return joblib.load(path)
    elif model_type == "torch":
        if model_class is None:
            raise ValueError("For PyTorch, 'model_class' must be provided")
        model = model_class()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    else:
        raise ValueError("Unsupported model type for loading")


def save_scaler(scaler, path: str):
    """Saves scaler."""
    joblib.dump(scaler, path)


def load_scaler(path: str):
    """Loads saved scaler."""
    return joblib.load(path)
