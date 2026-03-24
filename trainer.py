"""
training/trainer.py
-------------------
Generic Trainer for the XMTL model.

Handles:
    - Train / validation / test loops
    - Early stopping on validation MAE
    - ReduceLROnPlateau scheduler
    - Gradient clipping (max_norm=1.0)
    - Checkpoint save / load
    - Per-epoch metric logging

Usage:
    from training.trainer import Trainer
    trainer = Trainer(model, cfg, logger)
    trainer.fit(train_loader, val_loader)
    metrics = trainer.test(test_loader)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace
from typing import Dict, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from evaluation.metrics import evaluate_all


class Trainer:
    """XMTL model trainer with early stopping and LR scheduling.

    Args:
        model:      XMTL model instance.
        cfg:        Config namespace.
        logger:     Logger instance.
        device:     torch device.
        save_dir:   Directory for checkpoints and results.
    """
    def __init__(
        self,
        model: nn.Module,
        cfg: SimpleNamespace,
        logger,
        device: torch.device,
        save_dir: str,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5,
        )

        self.epochs = cfg.training.epochs
        self.patience = cfg.training.early_stopping_patience
        self.max_grad_norm = 1.0

        # Tracking
        self.best_val_mae = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_state = None
        self.history = []

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move all tensors in batch dict to device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch. Only tracks loss (no prediction accumulation).

        Returns:
            Dict with loss and approximate mae from loss.
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n = 0

        for batch in loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()

            out = self.model(batch)
            loss = self.criterion(out["prediction"], batch["label"])
            loss.backward()

            # Gradient clipping for GRU stability
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            with torch.no_grad():
                total_mae += (out["prediction"] - batch["label"]).abs().sum().item()
            n += bs

        return {"loss": total_loss / n, "mae": total_mae / n}

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one evaluation epoch (val or test).

        Returns:
            Dict with loss, mae, rmse, spearman, r2.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            batch = self._move_batch(batch)
            out = self.model(batch)
            loss = self.criterion(out["prediction"], batch["label"])

            total_loss += loss.item() * batch["label"].size(0)
            all_preds.append(out["prediction"].cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

        n = len(loader.dataset)
        preds = np.concatenate(all_preds).squeeze()
        labels = np.concatenate(all_labels).squeeze()
        metrics = evaluate_all(labels, preds)
        metrics["loss"] = total_loss / n
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Full training loop with early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
        """
        self.logger.info(f"Training for up to {self.epochs} epochs, "
                         f"patience={self.patience}")
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            train_m = self._train_epoch(train_loader)
            val_m = self._eval_epoch(val_loader)

            # LR scheduler step
            self.scheduler.step(val_m["mae"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Early stopping check
            if val_m["mae"] < self.best_val_mae:
                self.best_val_mae = val_m["mae"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Log
            record = {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_mae": train_m["mae"],
                "val_loss": val_m["loss"],
                "val_mae": val_m["mae"],
                "val_rmse": val_m["rmse"],
                "val_spearman": val_m["spearman"],
                "val_r2": val_m["r2"],
                "lr": current_lr,
                "patience": self.patience_counter,
            }
            self.history.append(record)

            if epoch % 5 == 0 or self.patience_counter == 0:
                self.logger.info(
                    f"  Epoch {epoch:3d}  "
                    f"train_mae={train_m['mae']:.4f}  "
                    f"val_mae={val_m['mae']:.4f}  "
                    f"val_r2={val_m['r2']:.4f}  "
                    f"lr={current_lr:.6f}  "
                    f"patience={self.patience_counter}/{self.patience}"
                )

            if self.patience_counter >= self.patience:
                self.logger.info(f"  Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t0
        self.logger.info(f"  Training done in {elapsed:.1f}s. "
                         f"Best epoch: {self.best_epoch}, "
                         f"best val MAE: {self.best_val_mae:.4f}")

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        # Save training history
        self._save_history()

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set using best model weights.

        Args:
            test_loader: Test DataLoader.

        Returns:
            Dict of test metrics.
        """
        metrics = self._eval_epoch(test_loader)
        self.logger.info(
            f"  TEST  mae={metrics['mae']:.4f}  rmse={metrics['rmse']:.4f}  "
            f"spearman={metrics['spearman']:.4f}  r2={metrics['r2']:.4f}"
        )

        # Save test metrics
        with open(os.path.join(self.save_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save test predictions
        self._save_predictions(test_loader)

        return metrics

    @torch.no_grad()
    def _save_predictions(self, loader: DataLoader) -> None:
        """Save test predictions to CSV."""
        import pandas as pd
        self.model.eval()
        all_preds, all_labels, all_weights = [], [], []

        for batch in loader:
            batch = self._move_batch(batch)
            out = self.model(batch)
            all_preds.append(out["prediction"].cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
            all_weights.append(out["modality_weights"].cpu().numpy())

        preds = np.concatenate(all_preds).squeeze()
        labels = np.concatenate(all_labels).squeeze()
        weights = np.concatenate(all_weights, axis=0)

        df = pd.DataFrame({"y_true": labels, "y_pred": preds})
        # Add modality weight columns
        modality_names = ["visual", "acoustic", "textual", "metadata", "creator", "temporal"]
        for i in range(weights.shape[1]):
            col_name = modality_names[i] if i < len(modality_names) else f"mod_{i}"
            df[f"w_{col_name}"] = weights[:, i]

        df.to_csv(os.path.join(self.save_dir, "test_predictions.csv"), index=False)

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_mae": self.best_val_mae,
        }
        path = os.path.join(self.save_dir, "best_model.pth" if is_best else "last_model.pth")
        torch.save(ckpt, path)

    def _save_history(self) -> None:
        """Save training history to CSV."""
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.save_dir, "training_history.csv"), index=False)
