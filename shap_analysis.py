import os
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    MODEL_SAVE_PATHS,
    SCALER_SAVE_PATH,
    FEATURES_TO_LOG,
    DESCRIPTOR_NAMES,
    CSV_PATH
)
from utils import (
    load_model,
    apply_log_transform,
    load_scaler
)

import matplotlib
matplotlib.use("Agg")


def load_features_for_analysis(csv_path: str, features_to_use, features_to_log: tuple):
    """
    Loads and preprocesses feature data: log-transforms and normalizes.

    Parameters:
        csv_path (str): Path to the CSV file with descriptors;
        features_to_use (tuple): Names of the features to use;
        features_to_log (tuple): Names of the features to log-transform.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with normalized features.
    """
    df = pd.read_csv(csv_path).drop_duplicates()
    feature_df = df[features_to_use].copy()
    feature_df = apply_log_transform(feature_df, features_to_log)
    scaler = load_scaler(SCALER_SAVE_PATH)
    normalized = scaler.transform(feature_df)
    return pd.DataFrame(normalized, columns=features_to_use)


def compute_shap_values(model, input_features: pd.DataFrame, model_type: str):
    """
    Computes SHAP values depending on model type.

    Parameters:
        model: Trained model;
        input_features (pd.DataFrame): Feature matrix;
        model_type (str): One of 'xgboost', 'random_forest', 'torch'.

    Returns:
        explainer, shap_values
    """
    if model_type == "random_forest":
        explainer = shap.Explainer(model, input_features, feature_perturbation="interventional")
        shap_values = explainer(input_features, check_additivity=False)
    elif model_type == "xgboost":
        explainer = shap.Explainer(model, input_features)
        shap_values = explainer(input_features)
    elif model_type == "torch":
        model.eval()
        def model_fn(x): return model(torch.from_numpy(x).float()).detach().numpy()
        explainer = shap.KernelExplainer(model_fn, input_features.values[:100])  # Background subset
        raw_values = explainer.shap_values(input_features.values[:100])  # shape (100, 8, 1)
        shap_values = np.squeeze(raw_values, axis=-1)  # shape (100, 8)
        input_features = input_features.iloc[:100]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return explainer, shap_values, input_features


def plot_summary(shap_values, input_features: pd.DataFrame, output_dir: str, model_name: str):
    """
    Saves SHAP summary plot.

    Parameters:
        shap_values: SHAP values;
        input_features (pd.DataFrame): Feature data;
        output_dir (str): Output directory;
        model_name (str): Model identifier.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, input_features, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_summary.png"))
    plt.close()


def plot_dependence(shap_values,
                    input_features: pd.DataFrame,
                    feature_name: str,
                    output_dir: str,
                    model_name: str):
    """
    Saves SHAP dependence plot for one feature.

    Parameters:
        shap_values: SHAP Explanation object or array (for older versions);
        input_features (pd.DataFrame): Feature data;
        feature_name (str): Feature to visualize;
        output_dir (str): Output directory;
        model_name (str): Model identifier.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()

    try:
        if hasattr(shap_values, 'values'):
            shap.dependence_plot(
                feature_name,
                shap_values.values,
                input_features.values,
                feature_names=input_features.columns.tolist(),
                show=False
            )
        else:
            shap.dependence_plot(
                feature_name,
                shap_values,
                input_features.values,
                feature_names=input_features.columns.tolist(),
                show=False
            )
    except Exception as e:
        print(f"[ERROR] Failed to plot dependence for {feature_name}: {e}")
        raise e

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_dependence_{feature_name}.png"))
    plt.close()


def analyze_model_shap(model_type: str, output_dir: str, top_n_features: int, model_class=None):
    """
    Unified SHAP analysis function for all supported models.

    Parameters:
        model_type (str): One of 'xgboost', 'random_forest', 'torch';
        output_dir (str): Directory to save output plots;
        top_n_features (int): Number of features to visualize in dependence plots;
        model_class: Required only for PyTorch models.
    """

    print(f"[INFO] Loading model: {model_type}")
    model = load_model(model_type, MODEL_SAVE_PATHS[model_type], model_class=model_class)
    input_features = load_features_for_analysis(CSV_PATH, DESCRIPTOR_NAMES, FEATURES_TO_LOG)

    print("[INFO] Computing SHAP values...")
    explainer, shap_values, input_features_limited = compute_shap_values(model, input_features, model_type)

    print("[INFO] Saving SHAP summary plot...")
    plot_summary(shap_values, input_features_limited, output_dir, model_name=model_type)

    print("[INFO] Saving SHAP dependence plots...")
    print("[INFO] Calculating top features for dependence plots...")

    if hasattr(shap_values, 'display_data') and shap_values.display_data is None:
        shap_values.display_data = input_features_limited.values

    if isinstance(shap_values, (list, np.ndarray)):
        values = np.array(shap_values)
    elif hasattr(shap_values, 'values'):
        values = shap_values.values
    else:
        raise ValueError(f"shap_values has unsupported type: {type(shap_values)} — missing .values attribute")

    mean_abs = np.abs(values).mean(axis=0).reshape(-1)
    indices = np.argsort(-mean_abs)[:top_n_features].astype(int)
    top_features = [DESCRIPTOR_NAMES[i] for i in indices]

    for feat in top_features:
        plot_dependence(shap_values, input_features_limited, feat, output_dir, model_name=model_type)

    print(f"[DONE] SHAP analysis completed for {model_type}. Plots saved to: {output_dir}")
