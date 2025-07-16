import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import (
    PERCEPTRON_INPUT_DIM,
    PERCEPTRON_HIDDEN_DIMS,
    PERCEPTRON_OUTPUT_DIM,
    PERCEPTRON_LEARNING_RATE,
    PERCEPTRON_EPOCHS,
    PERCEPTRON_BATCH_SIZE,
    PERCEPTRON_USE_SCHEDULER,
    PERCEPTRON_WEIGHT_DECAY,
    EARLY_STOPPING_MIN_DELTA,
    MODEL_SAVE_PATHS
)
from utils import save_model


class PerceptronRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(PERCEPTRON_INPUT_DIM, PERCEPTRON_HIDDEN_DIMS[0]),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(PERCEPTRON_HIDDEN_DIMS[0], PERCEPTRON_HIDDEN_DIMS[1]),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(PERCEPTRON_HIDDEN_DIMS[1], PERCEPTRON_OUTPUT_DIM)
        )

    def forward(self, x):
        return self.model(x)


def train_model(train_features, train_targets, val_features, val_targets):
    """
    Trains a perceptron-based regression model using PyTorch with support for:
        GPU acceleration (if available);
        Early stopping based on validation loss;
        Dynamic learning rate scheduling;
        TensorBoard logging (train/validation loss and learning rate).

    Parameters:
        train_features (np.ndarray): Normalized training feature matrix of shape (n_samples, n_features);
        train_targets (np.ndarray): Corresponding target values for training data of shape (n_samples,);
        val_features (np.ndarray): Normalized validation feature matrix used for early stopping and evaluation;
        val_targets (np.ndarray): Corresponding target values for validation data.

    Returns:
        model (torch.nn.Module): Trained PyTorch model with the best parameters (based on validation loss).
    """

    # --- Set device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Prepare tensors ---
    x_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_targets, dtype=torch.float32).view(-1, 1).to(device)
    x_val = torch.tensor(val_features, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_targets, dtype=torch.float32).view(-1, 1).to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=PERCEPTRON_BATCH_SIZE, shuffle=True)

    # --- Model, optimizer, scheduler ---
    model = PerceptronRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=PERCEPTRON_LEARNING_RATE, weight_decay=PERCEPTRON_WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) if PERCEPTRON_USE_SCHEDULER else None

    # --- TensorBoard setup ---
    writer = SummaryWriter(log_dir="runs/perceptron")

    # --- Early stopping parameters ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 10
    best_model_state = None

    for epoch in range(1, PERCEPTRON_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_preds = model(x_val)
            val_loss = criterion(val_preds, y_val).item()

        # --- Scheduler step ---
        if scheduler:
            scheduler.step(val_loss)

        # --- TensorBoard logging ---
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # --- Early stopping check ---
        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    writer.close()

    # --- Restoring best model ---
    if best_model_state:
        model.load_state_dict(best_model_state)

    # --- Saving model ---
    save_model(model, MODEL_SAVE_PATHS["torch"])
    return model
