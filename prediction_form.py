import numpy as np
import pandas as pd
import torch
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import Qt, QLocale

from config import DESCRIPTOR_NAMES, FEATURES_TO_LOG, MODEL_SAVE_PATHS
from utils import apply_log_transform, load_model, load_scaler
from models.perceptron_model import PerceptronRegressor


class PredictionForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solubility Prediction")
        self.setFixedSize(450, 400)

        self.inputs = {}
        layout = QVBoxLayout()

        # --- Title ---
        layout.addWidget(QLabel("Enter molecular descriptors:"))

        # --- Input fields for descriptors ---
        for name in DESCRIPTOR_NAMES:
            hbox = QHBoxLayout()
            label = QLabel(name)
            line_edit = QLineEdit()
            validator = QDoubleValidator()      # Numeric only
            validator.setLocale(QLocale("en_US"))
            line_edit.setValidator(validator)
            self.inputs[name] = line_edit
            hbox.addWidget(label)
            hbox.addWidget(line_edit)
            layout.addLayout(hbox)

        # --- Model selection dropdown ---
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["xgboost", "random_forest", "torch"])
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # --- Predict button ---
        self.predict_button = QPushButton("Predict")
        # noinspection PyUnresolvedReferences
        self.predict_button.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_button)

        # --- Result label ---
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def make_prediction(self):
        try:
            # --- Read input values ---
            values = []
            for name in DESCRIPTOR_NAMES:
                text = self.inputs[name].text()
                if not text:
                    raise ValueError(f"{name} is empty")
                values.append(float(text))

            feature_vector = np.array([values])  # shape (1, 8)

            # --- Apply log-transform to necessary features ---
            df = apply_log_transform(
                pd.DataFrame(feature_vector, columns=DESCRIPTOR_NAMES),
                FEATURES_TO_LOG
            )

            # --- Normalize features ---
            scaler = load_scaler("saved_models/scaler.pkl")
            normalized = scaler.transform(df)

            # --- Load model ---
            model_type = self.model_combo.currentText()
            if model_type == "torch":
                model = load_model("torch", MODEL_SAVE_PATHS["torch"], model_class=PerceptronRegressor)
                model.eval()
                with torch.no_grad():
                    tensor_input = torch.tensor(normalized, dtype=torch.float32)
                    prediction = model(tensor_input).item()
            else:
                model = load_model(model_type, MODEL_SAVE_PATHS[model_type])
                prediction = model.predict(normalized)[0]

            # --- Display result ---
            self.result_label.setText(f"Predicted Solubility: {prediction:.4f}")

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))
