import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout, QMessageBox, QScrollArea, QDialog
)
from PyQt6.QtGui import QMovie, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QSize

from data_preprocessing import get_normalized_data
from models.random_forest_model import train_model as train_rf
from models.xgboost_model import train_model as train_xgb
from models.perceptron_model import train_model as train_nn
from shap_analysis import analyze_model_shap
from prediction_form import PredictionForm
from evaluation import evaluate_model
from config import SHAP_OUTPUTS_SAVE_DIR, MODEL_SAVE_PATHS
from models.perceptron_model import PerceptronRegressor
from utils import load_model


class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    success = pyqtSignal(object)

    def __init__(self, func, on_success_data=None):
        super().__init__()
        self.func = func
        self.on_success_data = on_success_data

    def run(self):
        try:
            self.func()
            # noinspection PyUnresolvedReferences
            self.success.emit(self.on_success_data)
            # noinspection PyUnresolvedReferences
            self.finished.emit()
        except Exception as e:
            # noinspection PyUnresolvedReferences
            self.error.emit(e)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solubility Prediction App")
        self.setFixedSize(500, 700)

        layout = QVBoxLayout()

        # Spinner
        self.spinner = QLabel()
        self.spinner.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.movie = QMovie("preloader.gif")
        self.movie.setScaledSize(QSize(128, 128))
        # noinspection PyUnresolvedReferences
        self.movie.frameChanged.connect(self.spinner.repaint)
        self.spinner.setMovie(self.movie)
        self.movie.start()
        self.movie.stop()

        self.spinner.setVisible(False)
        layout.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Prepare Data
        self.prepare_button = QPushButton("Prepare Data")
        # noinspection PyUnresolvedReferences
        self.prepare_button.clicked.connect(self.prepare_data)
        layout.addWidget(self.prepare_button)

        # Train buttons
        self.rf_button = QPushButton("Train Random Forest")
        # noinspection PyUnresolvedReferences
        self.rf_button.clicked.connect(self.train_random_forest)
        layout.addWidget(self.rf_button)
        self.eval_rf_btn = None

        self.xgb_button = QPushButton("Train XGBoost")
        # noinspection PyUnresolvedReferences
        self.xgb_button.clicked.connect(self.train_xgboost)
        layout.addWidget(self.xgb_button)
        self.eval_xgb_btn = None

        self.nn_button = QPushButton("Train Neural Network")
        # noinspection PyUnresolvedReferences
        self.nn_button.clicked.connect(self.train_neural_net)
        layout.addWidget(self.nn_button)
        self.eval_nn_btn = None

        # SHAP
        self.shap_button = QPushButton("Run SHAP Analysis")
        # noinspection PyUnresolvedReferences
        self.shap_button.clicked.connect(self.run_shap)
        layout.addWidget(self.shap_button)

        self.show_shap_button = None
        self.shap_window = None

        # Predict form
        self.predict_button = QPushButton("Open Prediction Form")
        # noinspection PyUnresolvedReferences
        self.predict_button.clicked.connect(self.open_prediction_form)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)
        self.data = None
        self.predict_window = None

    def show_spinner(self, show=True):
        self.spinner.setVisible(show)
        if show:
            self.movie.start()
        else:
            self.movie.stop()

    def update_status(self, message, success=True):
        color = "green" if success else "red"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.status_label.setText(message)

    def _on_task_finished(self, success=True, error=None):
        self.show_spinner(False)
        if success:
            self.update_status("Successfully completed", success=True)
        else:
            self.update_status(f"Error: {str(error)}", success=False)
            QMessageBox.critical(self, "Execution Error", str(error))

    def safe_execute(self, func, on_success=None, on_success_data=None):
        def wrapper():
            self.show_spinner(True)
            self.status_label.setText("Processing...")
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")

            self.thread = QThread()
            self.worker = Worker(func, on_success_data=on_success_data)
            self.worker.moveToThread(self.thread)

            # noinspection PyUnresolvedReferences
            self.thread.started.connect(self.worker.run)
            # noinspection PyUnresolvedReferences
            self.worker.finished.connect(lambda: self._on_task_finished(success=True))
            # noinspection PyUnresolvedReferences
            self.worker.error.connect(lambda e: self._on_task_finished(success=False, error=e))

            # noinspection PyUnresolvedReferences
            self.worker.finished.connect(self.thread.quit)
            # noinspection PyUnresolvedReferences
            self.worker.error.connect(self.thread.quit)
            # noinspection PyUnresolvedReferences
            self.worker.finished.connect(self.worker.deleteLater)
            # noinspection PyUnresolvedReferences
            self.worker.error.connect(self.worker.deleteLater)
            # noinspection PyUnresolvedReferences
            self.thread.finished.connect(self.thread.deleteLater)

            if on_success is not None:
                # noinspection PyUnresolvedReferences
                self.worker.success.connect(on_success)

            self.thread.start()

        return wrapper

    def prepare_data(self):
        self.safe_execute(self._prepare_data)()

    def _prepare_data(self):
        self.data = get_normalized_data()

    def train_random_forest(self):
        self.safe_execute(
            self._train_rf,
            on_success=self._on_rf_trained,
            on_success_data=None
        )()

    def _train_rf(self):
        if not self.data:
            raise RuntimeError("Please prepare data first")
        train_rf(self.data["train_features"], self.data["train_targets"])

    def _on_rf_trained(self):
        self.eval_rf_btn = QPushButton("Evaluate Random Forest")
        # noinspection PyUnresolvedReferences
        self.eval_rf_btn.clicked.connect(lambda: self.evaluate("random_forest"))
        self.layout().addWidget(self.eval_rf_btn)

    def train_xgboost(self):
        self.safe_execute(
            self._train_xgb,
            on_success=self._on_xgb_trained,
            on_success_data=None
        )()

    def _train_xgb(self):
        if not self.data:
            raise RuntimeError("Please prepare data first")
        train_xgb(self.data["train_features"], self.data["train_targets"])

    def _on_xgb_trained(self):
        self.eval_xgb_btn = QPushButton("Evaluate XGBoost")
        # noinspection PyUnresolvedReferences
        self.eval_xgb_btn.clicked.connect(lambda: self.evaluate("xgboost"))
        self.layout().addWidget(self.eval_xgb_btn)

    def train_neural_net(self):
        self.safe_execute(
            self._train_nn,
            on_success=self._on_nn_trained,
            on_success_data=None
        )()

    def _train_nn(self):
        if not self.data:
            raise RuntimeError("Please prepare data first")
        train_nn(
            self.data["train_features"], self.data["train_targets"],
            self.data["val_features"], self.data["val_targets"]
        )

    def _on_nn_trained(self):
        self.eval_nn_btn = QPushButton("Evaluate Neural Network")
        # noinspection PyUnresolvedReferences
        self.eval_nn_btn.clicked.connect(lambda: self.evaluate("torch"))
        self.layout().addWidget(self.eval_nn_btn)

    def evaluate(self, model_type):
        self.safe_execute(
            lambda: self._evaluate(model_type),
            on_success=self._on_evaluated
        )()

    def _evaluate(self, model_type):
        path = MODEL_SAVE_PATHS[model_type]
        cls = PerceptronRegressor if model_type == "torch" else None
        model = load_model(model_type, path, model_class=cls)
        self._last_eval_result = evaluate_model(
            model,
            self.data["test_features"],
            self.data["test_targets"],
            model_type,
            normalize_input=False
        )
        self._last_eval_model_type = model_type

    def _on_evaluated(self, _=None):
        result = self._last_eval_result
        model_type = self._last_eval_model_type

        summary = f"\nMetric ({model_type})\nMSE: {result['MSE']:.4f}\nMAE: {result['MAE']:.4f}\nR2: {result['R2']:.4f}"
        QMessageBox.information(self, f"Evaluation - {model_type}", summary)

        fig = result["figure"]
        canvas = FigureCanvas(fig)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{model_type.upper()} - Evaluation Plot")
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        dialog.setLayout(layout)
        dialog.setMinimumSize(640, 640)
        dialog.exec()

    def run_shap(self):
        self.safe_execute(
            self._run_shap,
            on_success=self._on_shap_ran,
            on_success_data=None
        )()

    @staticmethod
    def _run_shap():
        for model_type in ["xgboost", "random_forest", "torch"]:
            analyze_model_shap(
                model_type=model_type,
                output_dir=SHAP_OUTPUTS_SAVE_DIR,
                top_n_features=8,
                model_class=PerceptronRegressor if model_type == "torch" else None
            )

    def _on_shap_ran(self):
        self.show_shap_button = QPushButton("Show SHAP Plots")
        # noinspection PyUnresolvedReferences
        self.show_shap_button.clicked.connect(self.display_shap_plots)
        self.layout().addWidget(self.show_shap_button)

    def display_shap_plots(self):
        if not os.path.exists(SHAP_OUTPUTS_SAVE_DIR):
            QMessageBox.warning(self, "SHAP Plots", "No SHAP outputs found")
            return

        self.shap_window = QWidget()
        self.shap_window.setWindowTitle("SHAP Plots")
        self.shap_window.setMinimumSize(600, 800)

        layout = QVBoxLayout()

        scroll = QScrollArea()
        content = QWidget()
        content_layout = QVBoxLayout()

        for filename in os.listdir(SHAP_OUTPUTS_SAVE_DIR):
            if filename.endswith(".png"):
                label = QLabel(f"Plot: {filename}")
                image = QLabel()
                image.setPixmap(QPixmap(os.path.join(SHAP_OUTPUTS_SAVE_DIR, filename)).scaledToWidth(550))
                content_layout.addWidget(label)
                content_layout.addWidget(image)

        content.setLayout(content_layout)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.shap_window.setLayout(layout)
        self.shap_window.show()

    def open_prediction_form(self):
        self.predict_window = PredictionForm()
        self.predict_window.show()
