from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# --- File paths ---
CSV_PATH = "final_dataset.csv"
SCALER_SAVE_PATH = "saved_models/scaler.pkl"
MODEL_SAVE_PATHS = {
    "random_forest": "saved_models/random_forest.pkl",
    "xgboost": "saved_models/xgboost.pkl",
    "torch": "saved_models/perceptron.pth"
}
SHAP_OUTPUTS_SAVE_DIR = "shap_outputs"

# --- Names of features and target variable ---
DESCRIPTOR_NAMES = ['MolLogP', 'MolWt', 'TPSA', 'FractionCSP3',
                    'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'NumRings']
TARGET_COLUMN = "Experimental_Solubility"

# --- Features to which the log1p() function is applied ---
FEATURES_TO_LOG = ("MolWt", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds")

# --- Scalers for each feature ---
SCALERS_FOR_DESCRIPTORS = {
        'MolLogP': StandardScaler,
        'MolWt': RobustScaler,
        'TPSA': RobustScaler,
        'FractionCSP3': MinMaxScaler,
        'NumHAcceptors': StandardScaler,
        'NumHDonors': RobustScaler,
        'NumRotatableBonds': RobustScaler,
        'NumRings': StandardScaler
    }

# --- Data division ---
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.1765    # 85% × 0.1765 ≈ 15%
RANDOM_STATE = 42

# --- Grid of hyperparameters for selection via GridSearchCV ---
RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [50, 100],          # Number of trees in the forest
    "max_depth": [5, 10],               # Maximum tree depth
    "min_samples_split": [2, 5],        # Minimum number of samples required to split an internal node
    "min_samples_leaf": [2, 4]          # Minimum number of objects in a leaf
}

# --- Number of folds in cross-validation when choosing parameters (RandomForestRegressor) ---
RANDOM_FOREST_CV = 3

# --- Metric for assessing the quality of a model when selecting parameters ---
# A negative MSE value is used because GridSearchCV maximizes the metric
RANDOM_FOREST_SCORING = 'neg_mean_squared_error'

# --- Hyperparameters for fitting with GridSearchCV when training XGBoost ---
XGBOOST_HYPERPARAM_GRID = {
    "n_estimators": [50, 100],          # Number of trees in the ensemble
    "max_depth": [3, 5],                # Maximum depth of each tree (the deeper, the more complex the model)
    "learning_rate": [0.01, 0.1],       # Learning rate. Small values → more stable but slower learning
    "subsample": [0.6, 0.8],            # The proportion of the sample used to build each tree (to prevent overfitting)
}

# --- Number of folds in cross-validation when choosing parameters (XGBoost) ---
XGBOOST_CV = 5

# --- Hyperparameters for the neural network ---
PERCEPTRON_INPUT_DIM = len(DESCRIPTOR_NAMES)    # = 8
PERCEPTRON_HIDDEN_DIMS = [32, 16]
PERCEPTRON_OUTPUT_DIM = 1
PERCEPTRON_LEARNING_RATE = 0.001
PERCEPTRON_EPOCHS = 200
PERCEPTRON_BATCH_SIZE = 64
PERCEPTRON_USE_SCHEDULER = True
PERCEPTRON_WEIGHT_DECAY = 1e-4                  # For L2 regularization
EARLY_STOPPING_MIN_DELTA = 1e-4
