"""
experiments/ablation.py
-----------------------
Ablation study for the XMTL model.

Two study types:
    1. Leave-one-out: Remove one modality at a time, measure performance drop.
    2. Single-modality: Keep only one modality, measure standalone contribution.

Uses the XGBoost baseline for fast ablation (tree-based, no GPU needed),
and optionally the XMTL model if a checkpoint is available.

Usage:
    cd LTF_code/
    python -m experiments.ablation --config configs/default.yaml
    python -m experiments.ablation --config configs/default.yaml --xmtl_checkpoint outputs/results/xmtl_full/best_model.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from copy import deepcopy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import get_logger
from evaluation.metrics import evaluate_all

MODALITIES = ["visual", "acoustic", "textual", "metadata", "creator", "temporal"]

# Content feature slice boundaries (must match dataset.py)
CONTENT_SLICES = {
    "visual":   (0, 4096),
    "acoustic": (4096, 4096 + 2688),
    "textual":  (4096 + 2688, 4096 + 2688 + 1538),
    "metadata": (4096 + 2688 + 1538, 4096 + 2688 + 1538 + 22),
    "creator":  (4096 + 2688 + 1538 + 22, 8353),
}

# ============================================================================
# XGBoost-based ablation (fast, no GPU)
# ============================================================================

def _load_data(cfg):
    """Load content features and temporal target arrays."""
    dataset_dir = cfg.data.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.normpath(os.path.join(_PROJECT_DIR, dataset_dir))

    content = np.load(
        os.path.join(dataset_dir, cfg.data.content_features_file),
        allow_pickle=True,
    ).astype(np.float32)
    temporal = np.load(
        os.path.join(dataset_dir, cfg.data.temporal_target_file),
        allow_pickle=True,
    ).astype(np.float32)
    return content, temporal


def _load_splits(cfg):
    """Load pre-computed train/val/test split indices."""
    processed_dir = cfg.data.processed_dir
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(_PROJECT_DIR, processed_dir)
    seed = cfg.training.seed
    data = np.load(os.path.join(processed_dir, f"splits_seed{seed}.npz"))
    return data["train"], data["val"], data["test"]


def _build_features(content, temporal, mask: Dict[str, bool]):
    """Build flattened feature matrix with modality masking.

    Args:
        content: [N, 8353] content features.
        temporal: [N, 310] temporal + target.
        mask: Dict of modality_name -> bool (True = keep).

    Returns:
        X [N, D], y [N].
    """
    parts = []
    for name, (start, end) in CONTENT_SLICES.items():
        if mask.get(name, True):
            parts.append(content[:, start:end])

    # Temporal features (all columns except last = label)
    if mask.get("temporal", True):
        parts.append(temporal[:, :-1])

    if not parts:
        # Edge case: all modalities masked — return zeros
        X = np.zeros((content.shape[0], 1), dtype=np.float32)
    else:
        X = np.concatenate(parts, axis=1)

    y = temporal[:, -1]
    return X, y


def _train_xgboost(X_train, y_train, X_test, y_test, logger) -> Dict[str, float]:
    """Train XGBoost and return test metrics."""
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbosity=0,
    )
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )
    y_pred = model.predict(X_test_s)
    return evaluate_all(y_test, y_pred)


def run_xgboost_ablation(cfg, logger) -> pd.DataFrame:
    """Run leave-one-out and single-modality ablation with XGBoost.

    Returns:
        DataFrame with columns: study, modality, mae, rmse, spearman, r2.
    """
    content, temporal = _load_data(cfg)
    train_idx, _, test_idx = _load_splits(cfg)

    # Full model (all modalities)
    full_mask = {m: True for m in MODALITIES}
    X_train, y_train = _build_features(content[train_idx], temporal[train_idx], full_mask)
    X_test, y_test = _build_features(content[test_idx], temporal[test_idx], full_mask)

    logger.info("Training full XGBoost (all modalities)...")
    full_metrics = _train_xgboost(X_train, y_train, X_test, y_test, logger)
    logger.info(f"  Full: MAE={full_metrics['mae']:.4f} R2={full_metrics['r2']:.4f}")

    rows = [{"study": "full", "modality": "all", **full_metrics}]

    # Leave-one-out
    logger.info("\n--- Leave-One-Out Ablation ---")
    for mod in MODALITIES:
        mask = {m: True for m in MODALITIES}
        mask[mod] = False
        X_tr, y_tr = _build_features(content[train_idx], temporal[train_idx], mask)
        X_te, y_te = _build_features(content[test_idx], temporal[test_idx], mask)
        metrics = _train_xgboost(X_tr, y_tr, X_te, y_te, logger)
        drop = full_metrics["r2"] - metrics["r2"]
        logger.info(f"  w/o {mod:12s}: MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} (drop={drop:+.4f})")
        rows.append({"study": "leave_one_out", "modality": mod, **metrics})

    # Single-modality
    logger.info("\n--- Single-Modality Ablation ---")
    for mod in MODALITIES:
        mask = {m: False for m in MODALITIES}
        mask[mod] = True
        X_tr, y_tr = _build_features(content[train_idx], temporal[train_idx], mask)
        X_te, y_te = _build_features(content[test_idx], temporal[test_idx], mask)
        metrics = _train_xgboost(X_tr, y_tr, X_te, y_te, logger)
        logger.info(f"  only {mod:12s}: MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")
        rows.append({"study": "single_modality", "modality": mod, **metrics})

    return pd.DataFrame(rows)


# ============================================================================
# XMTL-based ablation (optional, requires checkpoint)
# ============================================================================

def run_xmtl_ablation(cfg, checkpoint_path: str, logger) -> pd.DataFrame:
    """Run leave-one-out ablation with the XMTL model using modality masking.

    Args:
        cfg: Config namespace.
        checkpoint_path: Path to trained XMTL checkpoint.
        logger: Logger instance.

    Returns:
        DataFrame with columns: study, modality, mae, rmse, spearman, r2.
    """
    import torch
    from src.dataloader import build_dataloaders
    from models.xmtl import build_xmtl
    from training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    # Full model
    logger.info("XMTL full model evaluation...")
    _, _, test_dl = build_dataloaders(cfg)
    model = build_xmtl(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    trainer = Trainer(model, cfg, logger, device, save_dir="/tmp/xmtl_ablation")
    full_metrics = trainer._eval_epoch(test_dl)
    logger.info(f"  Full: MAE={full_metrics['mae']:.4f} R2={full_metrics['r2']:.4f}")
    rows.append({"study": "xmtl_full", "modality": "all", **{k: full_metrics[k] for k in ["mae", "rmse", "spearman", "r2"]}})

    # Leave-one-out with modality masking
    logger.info("\n--- XMTL Leave-One-Out (modality masking) ---")
    content_modalities = ["visual", "acoustic", "textual", "metadata", "creator"]
    for mod in content_modalities:
        mask = {mod: False}
        _, _, test_dl_masked = build_dataloaders(cfg, modality_mask=mask)

        # Reload model for clean eval
        model = build_xmtl(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        trainer = Trainer(model, cfg, logger, device, save_dir="/tmp/xmtl_ablation")
        metrics = trainer._eval_epoch(test_dl_masked)
        drop = full_metrics["r2"] - metrics["r2"]
        logger.info(f"  w/o {mod:12s}: MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} (drop={drop:+.4f})")
        rows.append({"study": "xmtl_leave_one_out", "modality": mod,
                      **{k: metrics[k] for k in ["mae", "rmse", "spearman", "r2"]}})

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def main(config_path: str, xmtl_checkpoint: Optional[str] = None):
    cfg = load_config(config_path)
    set_seed(cfg.training.seed)
    logger = get_logger("ablation", log_dir=cfg.output.log_dir)

    result_dir = cfg.output.result_dir
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(_PROJECT_DIR, result_dir)
    save_dir = os.path.join(result_dir, "ablation")
    os.makedirs(save_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Ablation Study")
    logger.info("=" * 60)

    # XGBoost ablation (always runs)
    xgb_df = run_xgboost_ablation(cfg, logger)
    xgb_df.to_csv(os.path.join(save_dir, "xgboost_ablation.csv"), index=False)
    logger.info(f"\nXGBoost ablation saved to {save_dir}/xgboost_ablation.csv")

    # XMTL ablation (optional)
    if xmtl_checkpoint and os.path.isfile(xmtl_checkpoint):
        xmtl_df = run_xmtl_ablation(cfg, xmtl_checkpoint, logger)
        xmtl_df.to_csv(os.path.join(save_dir, "xmtl_ablation.csv"), index=False)
        logger.info(f"XMTL ablation saved to {save_dir}/xmtl_ablation.csv")

    logger.info("=" * 60)
    logger.info("Ablation study complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XMTL Ablation Study")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--xmtl_checkpoint", type=str, default=None,
                        help="Optional: path to XMTL checkpoint for model-based ablation")
    args = parser.parse_args()
    main(args.config, xmtl_checkpoint=args.xmtl_checkpoint)
