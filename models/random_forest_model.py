from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from config import (
    RANDOM_FOREST_PARAM_GRID,
    RANDOM_FOREST_CV,
    RANDOM_FOREST_SCORING,
    MODEL_SAVE_PATHS,
    RANDOM_STATE
)
from utils import save_model


def train_model(train_features, train_targets, _val_features=None, _val_targets=None):
    """
    Trains a RandomForestRegressor model with hyperparameter selection and saves the best model.

    Parameters:
        train_features (np.ndarray): Features of the training set;
        train_targets (np.ndarray): Target variable of the training set;
        _val_features (np.ndarray, optional): Not used;
        _val_targets (np.ndarray, optional): Not used.

    Returns:
        RandomForestRegressor: Trained best model.
    """

    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid=RANDOM_FOREST_PARAM_GRID,
        cv=RANDOM_FOREST_CV,
        scoring=RANDOM_FOREST_SCORING,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(train_features, train_targets)
    best_model = grid.best_estimator_

    # Saving the best model
    save_model(best_model, MODEL_SAVE_PATHS["random_forest"])

    return best_model
