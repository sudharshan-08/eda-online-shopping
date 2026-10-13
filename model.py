"""
Deep Learning Model Module for Online Shoppers Purchasing Intention Dataset
Implements a PyTorch Multilayer Perceptron (ShoppersMLP) for Tabular Binary Classification,
alongside baseline classifiers (Logistic Regression, Random Forest).
Includes early stopping, class-weighted BCE loss, AdamW optimizer, and model persistence.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Tuple, Optional, List


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for full reproducibility across NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ShoppersMLP(nn.Module):
    """
    Multilayer Perceptron (MLP) tailored for tabular e-commerce purchasing intention prediction.
    Features:
    - Linear hidden layers (128 -> 64 -> 32)
    - Batch Normalization for internal covariate shift stabilization
    - ReLU activations
    - Dropout regularizers (0.3 and 0.2) to prevent overfitting
    - Single raw logit output for numerical stability with BCEWithLogitsLoss
    """
    def __init__(self, in_features: int, hidden_dim1: int = 128, hidden_dim2: int = 64, hidden_dim3: int = 32, dropout_rate: float = 0.3):
        super(ShoppersMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),

            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),

            nn.Linear(hidden_dim3, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def train_shoppers_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 128,
    epochs: int = 30,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 6,
    seed: int = 42,
    save_model_path: Optional[str] = "outputs/reports/best_mlp_model.pt"
) -> Tuple[ShoppersMLP, Dict[str, List[float]]]:
    """
    Trains the ShoppersMLP model using mini-batch gradient descent with:
    - BCEWithLogitsLoss with positive class weight (pos_weight) for class imbalance
    - AdamW optimizer
    - ReduceLROnPlateau learning rate scheduler
    - Early stopping monitored on validation loss
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_features = X_train.shape[1]
    model = ShoppersMLP(in_features=in_features).to(device)

    # Class weighting to counter imbalanced ratio (~5.46:1)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                running_val_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(epoch_val_loss)

        # Log history
        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["lr"].append(current_lr)

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save to disk
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "in_features": in_features,
            "best_val_loss": best_val_loss,
            "history": history
        }, save_model_path)

    return model, history


def predict_shoppers_mlp(model: ShoppersMLP, X: np.ndarray) -> np.ndarray:
    """Computes predicted purchase probabilities (0.0 to 1.0) for input features X."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(tensor_x)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains standard baseline machine learning models (Logistic Regression & Random Forest)
    for comparative benchmarking against the deep learning MLP.
    """
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        C=1.0
    )
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return {
        "logistic_regression": log_reg,
        "random_forest": rf
    }
