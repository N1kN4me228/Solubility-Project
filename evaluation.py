import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import SCALER_SAVE_PATH, FEATURES_TO_LOG
from utils import apply_log_transform, load_scaler


def evaluate_model(model, feature_table, solubility_values, model_type, normalize_input=False):
    """
    Evaluates the model by MSE, MAE, and R². Optionally normalizes the input data
    and plots the predicted values against the actual values.

    Parameters:
        model: the trained model (RandomForest, XGBoost, or neural network);
        feature_table: The feature matrix;
        solubility_values: The true solubility values;
        model_type: 'random_forest', 'xgboost' or 'torch';
        normalize_input: If True, applies the saved scaler to the feature_table.

    Returns:
        A dictionary with metrics.
    """

    if normalize_input:
        scaler = load_scaler(SCALER_SAVE_PATH)
        feature_table = scaler.transform(apply_log_transform(feature_table, FEATURES_TO_LOG))

    if model_type == "torch":
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(feature_table, dtype=torch.float32)).squeeze().numpy()
    else:
        predictions = model.predict(feature_table)

    # Metrics
    mse = mean_squared_error(solubility_values, predictions)
    mae = mean_absolute_error(solubility_values, predictions)
    r2 = r2_score(solubility_values, predictions)

    # "Prediction vs. Actual" plot
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(solubility_values, predictions, alpha=0.5, color="blue")
    plt.plot([min(solubility_values), max(solubility_values)],
             [min(solubility_values), max(solubility_values)],
             color="red", linestyle="--")
    plt.xlabel("True Solubility")
    plt.ylabel("Predicted Solubility")
    plt.title(f"{model_type.upper()} - Prediction vs Actual")

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "figure": fig
    }
