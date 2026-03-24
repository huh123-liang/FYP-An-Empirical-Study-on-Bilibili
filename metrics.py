"""
evaluation/metrics.py
---------------------
Unified evaluation metrics for regression tasks.

All functions accept numpy arrays and return scalar floats.
Used by both baseline experiments and the main XMTL model.

Primary metrics (reported in paper):
    - MAE:      Mean Absolute Error
    - RMSE:     Root Mean Squared Error
    - Spearman: Spearman rank correlation coefficient
    - R²:       Coefficient of determination
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground truth values. Shape [n].
        y_pred: Predicted values. Shape [n].

    Returns:
        MAE as a float.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground truth values. Shape [n].
        y_pred: Predicted values. Shape [n].

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient.

    Measures how well the predicted ranking matches the true ranking.
    Uses the raw (log-scaled) values — ranking is invariant to monotonic transforms.

    Args:
        y_true: Ground truth values. Shape [n].
        y_pred: Predicted values. Shape [n].

    Returns:
        Spearman correlation as a float. Range [-1, 1].
    """
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²).

    Args:
        y_true: Ground truth values. Shape [n].
        y_pred: Predicted values. Shape [n].

    Returns:
        R² as a float. Can be negative if predictions are worse than mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all four primary metrics at once.

    Args:
        y_true: Ground truth values. Shape [n].
        y_pred: Predicted values. Shape [n].

    Returns:
        Dict with keys: mae, rmse, spearman, r2.
    """
    return {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "spearman": calculate_spearman(y_true, y_pred),
        "r2": calculate_r_squared(y_true, y_pred),
    }
