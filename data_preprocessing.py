import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from config import (
    CSV_PATH,
    TARGET_COLUMN,
    DESCRIPTOR_NAMES,
    FEATURES_TO_LOG,
    SCALERS_FOR_DESCRIPTORS,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    SCALER_SAVE_PATH
)
from utils import apply_log_transform, save_scaler


def get_normalized_data():
    """
    Loads data from CSV, applies log transformation to the desired features,
    normalizes the values and splits the data into training, validation and test sets.

    Returns:
        normalized_training_data;
        training_targets;
        normalized_validation_data;
        validation_targets;
        normalized_test_data;
        test_targets.
    """

    # --- Loading a table ---
    data_table = pd.read_csv(CSV_PATH)

    # --- Separation of features and target variable ---
    feature_table = data_table[DESCRIPTOR_NAMES].copy()
    solubility_values = data_table[TARGET_COLUMN].values

    # --- Logarithmic transformation for skewed features ---
    feature_table = apply_log_transform(feature_table, FEATURES_TO_LOG)

    # --- Division into test part (15%) ---
    temp_features, test_features, temp_targets, test_targets = train_test_split(
        feature_table, solubility_values, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --- Split the remainder into training (70%) and validation (15%) ---
    training_features, validation_features, training_targets, validation_targets = train_test_split(
        temp_features, temp_targets, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )

    # --- Setting up ColumnTransformer to normalize features ---
    transformers = []
    for name in DESCRIPTOR_NAMES:
        scaler_class = SCALERS_FOR_DESCRIPTORS[name]
        transformers.append((name, scaler_class(), [name]))

    combined_scaler = ColumnTransformer(transformers=transformers)

    # --- Training a scaler on training data ---
    combined_scaler.fit(training_features)

    # --- Applying scaling ---
    normalized_training_data = combined_scaler.transform(training_features)
    normalized_validation_data = combined_scaler.transform(validation_features)
    normalized_test_data = combined_scaler.transform(test_features)

    # --- Saving scaler ---
    save_scaler(combined_scaler, SCALER_SAVE_PATH)

    # --- Returning results ---
    return {
        "train_features": normalized_training_data,
        "train_targets": training_targets,
        "val_features": normalized_validation_data,
        "val_targets": validation_targets,
        "test_features": normalized_test_data,
        "test_targets": test_targets
    }
