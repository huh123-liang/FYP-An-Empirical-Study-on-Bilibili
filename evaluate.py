"""
evaluation/evaluate.py
----------------------
Standalone evaluation script for the XMTL project.

Primary mode (baseline-first):
    Scans result directories for predictions.csv files, recomputes all
    metrics from scratch, and generates a cross-model comparison table.
    Works without any trained XMTL model.

Secondary mode (checkpoint):
    Loads a saved XMTL checkpoint, runs inference on the test set, and
    saves predictions + metrics. Requires a trained model.

Usage:
    # From predictions (no model needed):
    cd LTF_code/
    python -m evaluation.evaluate --config configs/default.yaml

    # From checkpoint:
    python -m evaluation.evaluate --config configs/default.yaml \\
        --checkpoint outputs/results/xmtl_full/best_model.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from evaluation.metrics import evaluate_all
from utils.config import load_config
from utils.logger import get_logger


# ============================================================================
# Mode 1: Evaluate from existing predictions.csv files
# ============================================================================

def find_prediction_files(result_dir: str) -> List[Dict]:
    """Recursively find all predictions.csv files under result_dir.

    Returns:
        List of dicts: [{"name": "ridge_full", "path": "...", "dir": "..."}]
    """
    found = []
    for root, dirs, files in os.walk(result_dir):
        if "predictions.csv" in files:
            model_name = os.path.basename(root)
            found.append({
                "name": model_name,
                "path": os.path.join(root, "predictions.csv"),
                "dir": root,
            })
    return sorted(found, key=lambda x: x["name"])


def evaluate_from_predictions(pred_path: str) -> Dict[str, float]:
    """Load a predictions.csv and recompute all metrics.

    Expects columns: y_true, y_pred (index column optional).

    Returns:
        Dict with mae, rmse, spearman, r2.
    """
    df = pd.read_csv(pred_path)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    return evaluate_all(y_true, y_pred)


def evaluate_all_models(result_dir: str, logger) -> pd.DataFrame:
    """Scan result_dir, recompute metrics for every model, build comparison.

    Args:
        result_dir: Root results directory (e.g. outputs/results).
        logger:     Logger instance.

    Returns:
        DataFrame with columns: model, mae, rmse, spearman, r2.
    """
    pred_files = find_prediction_files(result_dir)
    if not pred_files:
        logger.warning(f"No predictions.csv found under {result_dir}")
        return pd.DataFrame()

    rows = []
    for entry in pred_files:
        try:
            metrics = evaluate_from_predictions(entry["path"])
            # Save per-model metrics.json (overwrite with fresh computation)
            with open(os.path.join(entry["dir"], "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            rows.append({"model": entry["name"], **metrics})
            logger.info(f"  {entry['name']:25s}  MAE={metrics['mae']:.4f}  "
                        f"RMSE={metrics['rmse']:.4f}  Spearman={metrics['spearman']:.4f}  "
                        f"R2={metrics['r2']:.4f}")
        except Exception as e:
            logger.error(f"  {entry['name']}: FAILED — {e}")
            rows.append({"model": entry["name"], "mae": None, "rmse": None,
                         "spearman": None, "r2": None})

    return pd.DataFrame(rows)


# ============================================================================
# Mode 2: Evaluate from checkpoint (optional, requires trained model)
# ============================================================================

def evaluate_from_checkpoint(
    checkpoint_path: str, cfg, logger, save_dir: str
) -> Dict[str, float]:
    """Load XMTL checkpoint, run test inference, save results.

    Args:
        checkpoint_path: Path to best_model.pth.
        cfg:             Config namespace.
        logger:          Logger instance.
        save_dir:        Directory to save predictions and metrics.

    Returns:
        Dict of test metrics.
    """
    import torch
    from src.dataloader import build_dataloaders
    from models.xmtl import build_xmtl
    from training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_dl = build_dataloaders(cfg)
    model = build_xmtl(cfg)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    os.makedirs(save_dir, exist_ok=True)
    trainer = Trainer(model, cfg, logger, device, save_dir)
    metrics = trainer.test(test_dl)
    return metrics


# ============================================================================
# Summary printer
# ============================================================================

def print_comparison(df: pd.DataFrame, logger):
    """Print a formatted comparison table."""
    logger.info("")
    logger.info("=" * 75)
    logger.info("MODEL COMPARISON (recomputed from predictions)")
    logger.info("=" * 75)
    header = f"{'Model':25s} {'MAE':>8s} {'RMSE':>8s} {'Spearman':>10s} {'R²':>8s}"
    logger.info(header)
    logger.info("-" * 75)
    for _, row in df.iterrows():
        if row["mae"] is not None:
            line = (f"{row['model']:25s} {row['mae']:>8.4f} {row['rmse']:>8.4f} "
                    f"{row['spearman']:>10.4f} {row['r2']:>8.4f}")
        else:
            line = f"{row['model']:25s} {'FAIL':>8s}"
        logger.info(line)
    logger.info("=" * 75)


# ============================================================================
# Main
# ============================================================================

def main(config_path: str, checkpoint: Optional[str] = None):
    cfg = load_config(config_path)
    logger = get_logger("evaluate", log_dir=cfg.output.log_dir)

    result_dir = cfg.output.result_dir
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(_PROJECT_DIR, result_dir)

    logger.info("=" * 60)
    logger.info("XMTL Evaluation")
    logger.info("=" * 60)

    # Mode 1: Recompute from all existing predictions
    logger.info("Scanning for predictions.csv files...")
    df = evaluate_all_models(result_dir, logger)

    if not df.empty:
        # Save comparison CSV
        comp_path = os.path.join(result_dir, "comparison.csv")
        df.to_csv(comp_path, index=False)
        logger.info(f"Comparison saved to: {comp_path}")
        print_comparison(df, logger)

    # Mode 2: Checkpoint evaluation (if provided)
    if checkpoint and os.path.isfile(checkpoint):
        logger.info(f"\nEvaluating from checkpoint: {checkpoint}")
        save_dir = os.path.join(result_dir, "xmtl_eval")
        metrics = evaluate_from_checkpoint(checkpoint, cfg, logger, save_dir)
        logger.info(f"Checkpoint eval: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XMTL Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: path to XMTL checkpoint for model evaluation")
    args = parser.parse_args()
    main(args.config, checkpoint=args.checkpoint)
