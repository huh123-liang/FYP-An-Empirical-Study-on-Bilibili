"""
experiments/run_baselines.py
----------------------------
Unified entry point for all baseline experiments.

Loads data from npy files, constructs flat feature vectors, trains each
baseline model, evaluates on the test set, and saves results.

Feature inputs:
    full_flat:    content_features (8353) + temporal_flat (309) = 8662 dims
    content_only: content_features (8353) only

Standardization strategy:
    Ridge / MLP  -> StandardScaler (fit on train, transform val/test)
    RF / XGBoost -> no scaler (tree models are scale-invariant)

Smoke test mode:
    --smoke_test flag uses a subset of training data (default 5000 samples)
    to quickly verify the pipeline before full training.

Usage:
    cd LTF_code/
    python -m experiments.run_baselines --config configs/default.yaml
    python -m experiments.run_baselines --config configs/default.yaml --smoke_test
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import get_logger
from evaluation.metrics import evaluate_all


# ============================================================================
# Data loading
# ============================================================================

def load_flat_data(cfg, smoke_test: bool = False, smoke_n: int = 5000) -> Dict:
    """Load and flatten all features into numpy arrays.

    Args:
        cfg:        Config namespace.
        smoke_test: If True, subsample training data to smoke_n samples.
        smoke_n:    Number of training samples for smoke test.

    Returns:
        Dict with keys:
            X_train_full [n_train, 8662], y_train_full [n_train]
            X_train_content [n_train, 8353], (content-only)
            X_val_full [n_val, 8662], y_val [n_val]
            X_val_content [n_val, 8353]
            X_test_full [n_test, 8662], y_test [n_test]
            X_test_content [n_test, 8353]
            test_indices [n_test]
            content_dim: int, full_dim: int
    """
    # Resolve paths
    dataset_dir = cfg.data.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.normpath(os.path.join(_PROJECT_DIR, dataset_dir))

    processed_dir = cfg.data.processed_dir
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(_PROJECT_DIR, processed_dir)

    # Load raw arrays
    content = np.load(
        os.path.join(dataset_dir, cfg.data.content_features_file),
        allow_pickle=True, mmap_mode="r",
    )  # [50791, 8353]

    temporal = np.load(
        os.path.join(dataset_dir, cfg.data.temporal_target_file),
        allow_pickle=True, mmap_mode="r",
    )  # [50791, 310]

    # Separate temporal features and label
    tl = cfg.temporal_layout
    temporal_feat_dim = (tl.short_steps * tl.short_features +
                         tl.long_steps * tl.long_features +
                         tl.t_p_dim)  # 309
    temporal_features = np.array(temporal[:, :temporal_feat_dim], dtype=np.float32)  # [n, 309]
    labels = np.array(temporal[:, -1], dtype=np.float32)                             # [n]
    content_flat = np.array(content, dtype=np.float32)                               # [n, 8353]

    # Full flat: [content (8353) | temporal (309)] = [n, 8662]
    X_full = np.concatenate([content_flat, temporal_features], axis=1)

    # Load splits
    seed = cfg.training.seed
    split_path = os.path.join(processed_dir, f"splits_seed{seed}.npz")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            f"Run `python -m src.preprocessing` first."
        )
    splits = np.load(split_path)
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # Smoke test: subsample training data
    if smoke_test and len(train_idx) > smoke_n:
        rng = np.random.RandomState(cfg.training.seed)
        train_idx = rng.choice(train_idx, size=smoke_n, replace=False)

    return {
        "X_train_full": X_full[train_idx],
        "X_train_content": content_flat[train_idx],
        "y_train": labels[train_idx],
        "X_val_full": X_full[val_idx],
        "X_val_content": content_flat[val_idx],
        "y_val": labels[val_idx],
        "X_test_full": X_full[test_idx],
        "X_test_content": content_flat[test_idx],
        "y_test": labels[test_idx],
        "test_indices": test_idx,
        "content_dim": content_flat.shape[1],
        "full_dim": X_full.shape[1],
        "n_train": len(train_idx),
    }


# ============================================================================
# Baseline runners — each returns metrics dict or raises on failure
# ============================================================================


def run_ridge(data: Dict, result_dir: str, logger, variant: str = "full") -> Dict[str, float]:
    """Train and evaluate Ridge Regression baseline.

    Args:
        data:       Data dict from load_flat_data().
        result_dir: Base result directory.
        logger:     Logger instance.
        variant:    'full' (8662 dims) or 'content_only' (8353 dims).

    Returns:
        Dict of test metrics.
    """
    from sklearn.linear_model import Ridge
    from models.baseline_models import SklearnBaseline

    suffix = "ridge_content_only" if variant == "content_only" else "ridge_full"
    X_key = "X_train_content" if variant == "content_only" else "X_train_full"
    Xv_key = "X_val_content" if variant == "content_only" else "X_val_full"
    Xt_key = "X_test_content" if variant == "content_only" else "X_test_full"

    logger.info("=" * 50)
    logger.info(f"Training {suffix} (input_dim={data[X_key].shape[1]})...")

    # Alpha grid search on val set
    best_alpha, best_score = None, float("inf")
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        model.fit(data[X_key], data["y_train"])
        val_pred = model.predict(data[Xv_key])
        val_mae = float(np.mean(np.abs(data["y_val"] - val_pred)))
        logger.info(f"  alpha={alpha:7.1f}  val_mae={val_mae:.4f}")
        if val_mae < best_score:
            best_score = val_mae
            best_alpha = alpha

    logger.info(f"  Best alpha: {best_alpha}")
    final_model = Ridge(alpha=best_alpha)
    save_dir = os.path.join(result_dir, suffix)
    baseline = SklearnBaseline(suffix, final_model, save_dir, use_scaler=True)
    baseline.train(data[X_key], data["y_train"])
    metrics = baseline.evaluate_and_save(
        data[Xt_key], data["y_test"], data["test_indices"],
        config_dict={
            "model": suffix, "alpha": best_alpha,
            "input_dim": data[X_key].shape[1], "variant": variant,
        },
    )
    logger.info(f"  {suffix} test: {metrics}")
    return metrics


def run_random_forest(data: Dict, result_dir: str, logger) -> Dict[str, float]:
    """Train and evaluate Random Forest baseline.

    Uses constrained parameters to handle 8662-dim input:
        n_estimators=100, max_depth=15, max_features=0.3
    """
    from sklearn.ensemble import RandomForestRegressor
    from models.baseline_models import SklearnBaseline

    logger.info("=" * 50)
    logger.info("Training Random Forest baseline...")

    rf_params = {
        "n_estimators": 100,
        "max_depth": 15,
        "max_features": 0.3,
        "n_jobs": -1,
        "random_state": 42,
    }
    model = RandomForestRegressor(**rf_params)
    save_dir = os.path.join(result_dir, "random_forest")
    baseline = SklearnBaseline("random_forest", model, save_dir, use_scaler=False)
    baseline.train(data["X_train_full"], data["y_train"])
    metrics = baseline.evaluate_and_save(
        data["X_test_full"], data["y_test"], data["test_indices"],
        config_dict={"model": "random_forest", **rf_params, "input_dim": data["full_dim"]},
    )
    logger.info(f"  RF test: {metrics}")
    return metrics


def run_xgboost(data: Dict, result_dir: str, logger) -> Dict[str, float]:
    """Train and evaluate XGBoost baseline with early stopping."""
    import xgboost as xgb
    from models.baseline_models import SklearnBaseline

    logger.info("=" * 50)
    logger.info("Training XGBoost baseline...")

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "early_stopping_rounds": 20,
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**xgb_params)
    save_dir = os.path.join(result_dir, "xgboost")
    baseline = SklearnBaseline("xgboost", model, save_dir, use_scaler=False)
    baseline.train(data["X_train_full"], data["y_train"],
                   data["X_val_full"], data["y_val"])
    metrics = baseline.evaluate_and_save(
        data["X_test_full"], data["y_test"], data["test_indices"],
        config_dict={"model": "xgboost", **xgb_params, "input_dim": data["full_dim"]},
    )
    logger.info(f"  XGBoost test: {metrics}")
    return metrics


def run_mlp(data: Dict, result_dir: str, cfg, logger) -> Dict[str, float]:
    """Train and evaluate MLP baseline with StandardScaler."""
    from models.baseline_models import MLPBaseline

    logger.info("=" * 50)
    logger.info("Training MLP baseline...")

    device = "cpu"
    if cfg.training.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif cfg.training.device != "cpu":
        device = cfg.training.device

    save_dir = os.path.join(result_dir, "mlp")
    mlp = MLPBaseline(
        input_dim=data["full_dim"],
        save_dir=save_dir,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        patience=cfg.training.early_stopping_patience,
        batch_size=cfg.training.batch_size,
        dropout=cfg.model.dropout,
        device=device,
    )
    mlp.train(data["X_train_full"], data["y_train"],
              data["X_val_full"], data["y_val"])
    metrics = mlp.evaluate_and_save(
        data["X_test_full"], data["y_test"], data["test_indices"],
        config_dict={
            "model": "mlp", "input_dim": data["full_dim"],
            "lr": cfg.training.lr, "epochs": cfg.training.epochs,
            "patience": cfg.training.early_stopping_patience,
            "dropout": cfg.model.dropout, "device": device,
        },
    )
    logger.info(f"  MLP test: {metrics}")
    return metrics


# ============================================================================
# Summary
# ============================================================================


def print_summary(all_results: Dict[str, Dict], logger) -> pd.DataFrame:
    """Print a formatted comparison table of all baseline results.

    Handles both successful runs (metrics dict) and failures (error string).
    """
    logger.info("")
    logger.info("=" * 75)
    logger.info("BASELINE COMPARISON SUMMARY")
    logger.info("=" * 75)
    header = f"{'Model':<22s} {'MAE':>8s} {'RMSE':>8s} {'Spearman':>10s} {'R²':>8s} {'Time(s)':>8s} {'Status':>8s}"
    logger.info(header)
    logger.info("-" * 75)

    rows = []
    for name, entry in all_results.items():
        if entry["status"] == "ok":
            m = entry["metrics"]
            row = (f"{name:<22s} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
                   f"{m['spearman']:>10.4f} {m['r2']:>8.4f} "
                   f"{entry['time_sec']:>8.1f} {'ok':>8s}")
            rows.append({"model": name, **m, "time_sec": entry["time_sec"], "status": "ok"})
        else:
            row = f"{name:<22s} {'—':>8s} {'—':>8s} {'—':>10s} {'—':>8s} {'—':>8s} {'FAIL':>8s}"
            rows.append({"model": name, "mae": None, "rmse": None,
                         "spearman": None, "r2": None,
                         "time_sec": None, "status": "FAIL",
                         "error": entry.get("error", "")})
        logger.info(row)

    logger.info("=" * 75)
    return pd.DataFrame(rows)


def _run_one(name: str, fn, all_results: Dict, log, **kwargs):
    """Run a single baseline with timing and error isolation.

    Args:
        name:        Display name for this baseline.
        fn:          Runner function to call with **kwargs.
        all_results: Dict to store results into.
        log:         Logger instance (named 'log' to avoid collision with kwargs).

    On success, stores {"status": "ok", "metrics": {...}, "time_sec": float}.
    On failure, stores {"status": "fail", "error": str} and continues.
    """
    t0 = time.time()
    try:
        metrics = fn(**kwargs)
        elapsed = time.time() - t0
        all_results[name] = {"status": "ok", "metrics": metrics, "time_sec": elapsed}
        log.info(f"  {name} completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log.error(f"  {name} FAILED after {elapsed:.1f}s: {e}\n{tb}")
        all_results[name] = {"status": "fail", "error": str(e), "time_sec": elapsed}


# ============================================================================
# Main
# ============================================================================

def main(config_path: str, smoke_test: bool = False, smoke_n: int = 5000):
    cfg = load_config(config_path)
    set_seed(cfg.training.seed)
    logger = get_logger("baselines", log_dir=cfg.output.log_dir)

    result_dir = cfg.output.result_dir
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(_PROJECT_DIR, result_dir)
    result_dir = os.path.join(result_dir, "baselines")
    os.makedirs(result_dir, exist_ok=True)

    mode_str = f"SMOKE TEST (n={smoke_n})" if smoke_test else "FULL"
    logger.info(f"Mode: {mode_str}")
    logger.info("Loading flat data...")
    data = load_flat_data(cfg, smoke_test=smoke_test, smoke_n=smoke_n)
    logger.info(f"  X_train_full: {data['X_train_full'].shape}, "
                f"X_val_full: {data['X_val_full'].shape}, "
                f"X_test_full: {data['X_test_full'].shape}")
    logger.info(f"  X_train_content: {data['X_train_content'].shape}")
    logger.info(f"  Full dim: {data['full_dim']}, Content dim: {data['content_dim']}")

    all_results: Dict[str, Dict] = {}

    # 1. Ridge (content-only) — weakest static baseline
    _run_one("Ridge_content", run_ridge, all_results, logger,
             data=data, result_dir=result_dir, logger=logger, variant="content_only")

    # 2. Ridge (full) — static + temporal linear baseline
    _run_one("Ridge_full", run_ridge, all_results, logger,
             data=data, result_dir=result_dir, logger=logger, variant="full")

    # 3. MLP
    _run_one("MLP", run_mlp, all_results, logger,
             data=data, result_dir=result_dir, cfg=cfg, logger=logger)

    # 4. Random Forest
    _run_one("RandomForest", run_random_forest, all_results, logger,
             data=data, result_dir=result_dir, logger=logger)

    # 5. XGBoost
    _run_one("XGBoost", run_xgboost, all_results, logger,
             data=data, result_dir=result_dir, logger=logger)

    # Summary
    summary_df = print_summary(all_results, logger)
    summary_path = os.path.join(result_dir, "baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XMTL Baseline Experiments")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run with subset of training data for quick pipeline verification",
    )
    parser.add_argument(
        "--smoke_n", type=int, default=5000,
        help="Number of training samples for smoke test (default: 5000)",
    )
    args = parser.parse_args()
    main(args.config, smoke_test=args.smoke_test, smoke_n=args.smoke_n)
