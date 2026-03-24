"""
models/baseline_models.py
-------------------------
Baseline model definitions for XMTL experiments.

Contains:
    - SklearnBaseline:  Wrapper for sklearn / xgboost regressors with a
                        unified train/predict/evaluate interface.
                        Supports optional StandardScaler preprocessing.
    - MLPBaselineModel: A 3-layer MLP in PyTorch (512 -> 256 -> 1).
    - MLPBaseline:      Training wrapper with early stopping on validation loss.

All baselines operate on flat feature vectors:
    X = [content_features (8353) | temporal_flat (309)] = 8662 dims
    y = label (1 dim, continuous, log-scaled)

Standardization strategy:
    - Ridge / MLP: StandardScaler (fit on train only)
    - RF / XGBoost: no scaler (tree models are scale-invariant)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from evaluation.metrics import evaluate_all


# ============================================================================
# Sklearn / XGBoost Wrapper
# ============================================================================
class SklearnBaseline:
    """Unified wrapper for sklearn-style regressors.

    Args:
        name:       Baseline name (e.g. 'ridge', 'random_forest', 'xgboost').
        model:      A fitted or unfitted sklearn-compatible regressor instance.
        save_dir:   Directory to save results.
        use_scaler: If True, fit a StandardScaler on training data and apply
                    to all inputs. Scaler is fit only on train set.
    """

    def __init__(self, name: str, model: Any, save_dir: str, use_scaler: bool = False):
        self.name = name
        self.model = model
        self.save_dir = save_dir
        self.use_scaler = use_scaler
        self.scaler: Optional[StandardScaler] = None
        os.makedirs(save_dir, exist_ok=True)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the model, optionally with scaler preprocessing.

        Args:
            X_train: Training features. Shape [n_train, d].
            y_train: Training labels. Shape [n_train].
            X_val:   Validation features (optional). Shape [n_val, d].
            y_val:   Validation labels (optional). Shape [n_val].
        """
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        if self.name == "xgboost" and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions. Shape [n]."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate_and_save(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_indices: np.ndarray,
        config_dict: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Evaluate on test set and save metrics + predictions.

        Args:
            X_test:       Test features. Shape [n_test, d].
            y_test:       Test labels. Shape [n_test].
            test_indices: Original sample indices for traceability.
            config_dict:  Optional config dict to save alongside results.

        Returns:
            Dict of metrics: {mae, rmse, spearman, r2}.
        """
        y_pred = self.predict(X_test)
        metrics = evaluate_all(y_test, y_pred)

        # Save metrics
        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save predictions
        pred_df = pd.DataFrame({
            "index": test_indices,
            "y_true": y_test,
            "y_pred": y_pred,
        })
        pred_df.to_csv(os.path.join(self.save_dir, "predictions.csv"), index=False)

        # Save config (include scaler info)
        if config_dict is None:
            config_dict = {}
        config_dict["use_scaler"] = self.use_scaler
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        return metrics


# ============================================================================
# MLP Baseline (PyTorch)
# ============================================================================
class MLPBaselineModel(nn.Module):
    """3-layer MLP for regression on flat features.

    Architecture:
        input_dim -> 512 -> ReLU -> Dropout -> 256 -> ReLU -> Dropout -> 1

    Args:
        input_dim: Dimension of the flat input vector.
        dropout:   Dropout probability. Default 0.2.
    """

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),    # [B, input_dim] -> [B, 512]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),          # [B, 512] -> [B, 256]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),            # [B, 256] -> [B, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Flat input tensor. Shape [B, input_dim].

        Returns:
            Predictions. Shape [B, 1].
        """
        return self.net(x)


class MLPBaseline:
    """Training wrapper for MLPBaselineModel with early stopping.

    Handles the full train -> validate -> test -> save cycle.
    Applies StandardScaler to inputs (fit on train only).

    Args:
        input_dim:  Dimension of flat input vector.
        save_dir:   Directory to save results.
        lr:         Learning rate. Default 1e-3.
        epochs:     Max training epochs. Default 100.
        patience:   Early stopping patience. Default 15.
        batch_size: Training batch size. Default 128.
        dropout:    Dropout probability. Default 0.2.
        device:     'cpu' or 'cuda'. Default 'cpu'.
    """

    def __init__(
        self,
        input_dim: int,
        save_dir: str,
        lr: float = 1e-3,
        epochs: int = 100,
        patience: int = 15,
        batch_size: int = 128,
        dropout: float = 0.2,
        device: str = "cpu",
    ):
        self.name = "mlp"
        self.save_dir = save_dir
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.use_scaler = True
        self.scaler: Optional[StandardScaler] = None
        os.makedirs(save_dir, exist_ok=True)

        self.model = MLPBaselineModel(input_dim, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)  # [n, 1]
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train with early stopping on validation MSE.

        Applies StandardScaler (fit on train only).

        Args:
            X_train: Training features. Shape [n_train, d].
            y_train: Training labels. Shape [n_train].
            X_val:   Validation features. Shape [n_val, d].
            y_val:   Validation labels. Shape [n_val].
        """
        # Standardize
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X_batch)       # [B, 1]
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = self.model(X_batch)
                    val_loss += self.criterion(pred, y_batch).item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter == 0:
                print(f"  [MLP] Epoch {epoch:3d}  train_mse={train_loss:.4f}  "
                      f"val_mse={val_loss:.4f}  patience={patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                print(f"  [MLP] Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)

        # Save checkpoint
        torch.save(best_state, os.path.join(self.save_dir, "mlp_best.pth"))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions. Returns numpy array shape [n]."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            pred = self.model(X_t).cpu().numpy().squeeze(1)  # [n]
        return pred

    def evaluate_and_save(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_indices: np.ndarray,
        config_dict: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Evaluate on test set and save metrics + predictions.

        Returns:
            Dict of metrics: {mae, rmse, spearman, r2}.
        """
        y_pred = self.predict(X_test)
        metrics = evaluate_all(y_test, y_pred)

        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        pred_df = pd.DataFrame({
            "index": test_indices,
            "y_true": y_test,
            "y_pred": y_pred,
        })
        pred_df.to_csv(os.path.join(self.save_dir, "predictions.csv"), index=False)

        if config_dict is None:
            config_dict = {}
        config_dict["use_scaler"] = self.use_scaler
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        return metrics
