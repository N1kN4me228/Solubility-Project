from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from config import XGBOOST_HYPERPARAM_GRID, XGBOOST_CV, MODEL_SAVE_PATHS
from utils import save_model


def train_model(train_features, train_targets, _val_features=None, _val_targets=None):
    """
    Trains an XGBoost model using GridSearchCV.

    Parameters:
        train_features (np.ndarray): Features of the training set;
        train_targets (np.ndarray): Target variable of the training set;
        _val_features (np.ndarray, optional): Not used;
        _val_targets (np.ndarray, optional): Not used.

    Returns:
        The trained XGBRegressor model with the best parameters.
    """

    # --- Basic XGBoost model. Specify the task — regression (reg:squarederror) ---
    base_model = XGBRegressor(
        objective="reg:squarederror",
        verbosity=0,
        n_jobs=-1
    )

    # --- Hyperparameter enumeration using cross-validation ---
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=XGBOOST_HYPERPARAM_GRID,
        cv=XGBOOST_CV,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        verbose=0
    )

    # --- Model training ---
    grid_search.fit(train_features, train_targets)

    # --- Best model found ---
    best_model = grid_search.best_estimator_

    # --- Saving the best model ---
    save_model(best_model, MODEL_SAVE_PATHS["xgboost"])

    return best_model
