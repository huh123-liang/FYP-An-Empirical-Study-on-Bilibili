"""
explainability/shap_analysis.py
-------------------------------
SHAP-based explainability analysis for the XMTL project.

Two analysis modes:
    1. XGBoost SHAP: TreeExplainer on the best XGBoost baseline.
       Produces per-modality and per-feature importance scores.
    2. XMTL fusion weights: Analyze learned modality weights from
       the gated fusion layer (extracted from test predictions).

Usage:
    cd LTF_code/
    python -m explainability.shap_analysis --config configs/default.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import get_logger

MODALITIES = ["visual", "acoustic", "textual", "metadata", "creator"]
CONTENT_SLICES = {
    "visual":   (0, 4096),
    "acoustic": (4096, 4096 + 2688),
    "textual":  (4096 + 2688, 4096 + 2688 + 1538),
    "metadata": (4096 + 2688 + 1538, 4096 + 2688 + 1538 + 22),
    "creator":  (4096 + 2688 + 1538 + 22, 8353),
}
TEMPORAL_DIM = 309  # 72*4 + 6*3 + 3


# ============================================================================
# Data loading helpers
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
    processed_dir = cfg.data.processed_dir
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(_PROJECT_DIR, processed_dir)
    data = np.load(os.path.join(processed_dir, f"splits_seed{cfg.training.seed}.npz"))
    return data["train"], data["val"], data["test"]


def _build_full_features(content, temporal):
    """Build full feature matrix: content + temporal (no label)."""
    X = np.concatenate([content, temporal[:, :-1]], axis=1)
    y = temporal[:, -1]
    return X, y


def _feature_names():
    """Generate feature names for all dimensions."""
    names = []
    for mod, (start, end) in CONTENT_SLICES.items():
        for i in range(end - start):
            names.append(f"{mod}_{i}")
    # Temporal features
    for i in range(72 * 4):
        names.append(f"temporal_short_{i}")
    for i in range(6 * 3):
        names.append(f"temporal_long_{i}")
    for i in range(3):
        names.append(f"t_p_{i}")
    return names


# ============================================================================
# XGBoost SHAP analysis
# ============================================================================

def xgboost_shap_analysis(cfg, logger, save_dir: str) -> Dict:
    """Train XGBoost on full features and compute SHAP values.

    Returns:
        Dict with modality-level and top feature importances.
    """
    import shap
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    content, temporal = _load_data(cfg)
    train_idx, _, test_idx = _load_splits(cfg)

    X_train, y_train = _build_full_features(content[train_idx], temporal[train_idx])
    X_test, y_test = _build_full_features(content[test_idx], temporal[test_idx])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logger.info("Training XGBoost for SHAP analysis...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbosity=0,
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

    # SHAP TreeExplainer (fast for tree models)
    logger.info("Computing SHAP values (this may take a few minutes)...")
    # Use a subsample for speed
    n_explain = min(1000, len(X_test_s))
    X_explain = X_test_s[:n_explain]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)  # [n_explain, n_features]

    # Aggregate: mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_names = _feature_names()

    # Per-modality aggregation
    modality_importance = {}
    all_modalities = list(CONTENT_SLICES.keys()) + ["temporal"]
    idx = 0
    for mod in MODALITIES:
        start, end = CONTENT_SLICES[mod]
        dim = end - start
        modality_importance[mod] = float(mean_abs_shap[idx:idx + dim].sum())
        idx += dim
    # Temporal
    modality_importance["temporal"] = float(mean_abs_shap[idx:].sum())

    # Normalize to percentages
    total = sum(modality_importance.values())
    modality_pct = {k: v / total * 100 for k, v in modality_importance.items()}

    # Top 20 features
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]
    top_features = [
        {"feature": feat_names[i], "mean_abs_shap": float(mean_abs_shap[i])}
        for i in top_indices
    ]

    results = {
        "modality_importance": modality_importance,
        "modality_importance_pct": modality_pct,
        "top_20_features": top_features,
    }

    # Save
    with open(os.path.join(save_dir, "xgboost_shap.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save full SHAP values
    np.save(os.path.join(save_dir, "shap_values.npy"), shap_values)

    logger.info("\nModality importance (SHAP):")
    for mod in all_modalities:
        logger.info(f"  {mod:12s}: {modality_pct[mod]:6.2f}%")

    logger.info("\nTop 10 features:")
    for feat in top_features[:10]:
        logger.info(f"  {feat['feature']:30s}: {feat['mean_abs_shap']:.4f}")

    return results


# ============================================================================
# XMTL fusion weight analysis
# ============================================================================

def fusion_weight_analysis(predictions_csv: str, logger, save_dir: str) -> Dict:
    """Analyze modality weights from XMTL test predictions.

    Args:
        predictions_csv: Path to test_predictions.csv with w_* columns.
        logger: Logger instance.
        save_dir: Directory to save results.

    Returns:
        Dict with weight statistics.
    """
    df = pd.read_csv(predictions_csv)

    weight_cols = [c for c in df.columns if c.startswith("w_")]
    if not weight_cols:
        logger.warning("No modality weight columns found in predictions CSV.")
        return {}

    weights = df[weight_cols]
    modality_names = [c.replace("w_", "") for c in weight_cols]

    stats = {}
    logger.info("\nXMTL Fusion Weight Statistics:")
    logger.info(f"  {'Modality':12s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    logger.info("  " + "-" * 48)

    for col, name in zip(weight_cols, modality_names):
        s = {
            "mean": float(weights[col].mean()),
            "std": float(weights[col].std()),
            "min": float(weights[col].min()),
            "max": float(weights[col].max()),
            "median": float(weights[col].median()),
        }
        stats[name] = s
        logger.info(f"  {name:12s} {s['mean']:8.4f} {s['std']:8.4f} {s['min']:8.4f} {s['max']:8.4f}")

    # Correlation between weights and prediction error
    if "y_true" in df.columns and "y_pred" in df.columns:
        df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
        error_corr = {}
        for col, name in zip(weight_cols, modality_names):
            corr = float(df[col].corr(df["abs_error"]))
            error_corr[name] = corr
        stats["error_correlation"] = error_corr
        logger.info("\n  Weight-Error correlation:")
        for name, corr in error_corr.items():
            logger.info(f"    {name:12s}: {corr:+.4f}")

    results = {"fusion_weights": stats}
    with open(os.path.join(save_dir, "fusion_weight_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# Main
# ============================================================================

def main(config_path: str, predictions_csv: Optional[str] = None):
    cfg = load_config(config_path)
    set_seed(cfg.training.seed)
    logger = get_logger("shap_analysis", log_dir=cfg.output.log_dir)

    result_dir = cfg.output.result_dir
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(_PROJECT_DIR, result_dir)
    save_dir = os.path.join(result_dir, "explainability")
    os.makedirs(save_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Explainability Analysis")
    logger.info("=" * 60)

    # XGBoost SHAP
    xgb_results = xgboost_shap_analysis(cfg, logger, save_dir)

    # XMTL fusion weights (if predictions available)
    if predictions_csv is None:
        # Try to find test_predictions.csv automatically
        candidates = [
            os.path.join(result_dir, "xmtl_full", "test_predictions.csv"),
            os.path.join(result_dir, "xmtl_eval", "test_predictions.csv"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                predictions_csv = c
                break

    if predictions_csv and os.path.isfile(predictions_csv):
        logger.info(f"\nAnalyzing fusion weights from: {predictions_csv}")
        fusion_weight_analysis(predictions_csv, logger, save_dir)
    else:
        logger.info("\nNo XMTL predictions found — skipping fusion weight analysis.")

    logger.info("=" * 60)
    logger.info("Explainability analysis complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Explainability Analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to XMTL test_predictions.csv with modality weights")
    args = parser.parse_args()
    main(args.config, predictions_csv=args.predictions)
