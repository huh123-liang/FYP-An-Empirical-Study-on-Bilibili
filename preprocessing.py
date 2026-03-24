"""
src/preprocessing.py
--------------------
Data verification and preprocessing module.

Responsibilities:
  1. Verify that all required data files exist.
  2. Load npy files and check shapes against config expectations.
  3. Validate feature slicing dimensions (visual/acoustic/textual/metadata/social).
  4. Validate temporal layout (short/long/t_p/label).
  5. Generate and save train/val/test split indices for reproducibility.
  6. Report label statistics (range, mean, std, distribution).

This module MUST be run before training. It will raise clear errors if any
assumption is violated, rather than silently producing wrong results.

Usage:
    python -m src.preprocessing --config configs/default.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Allow running as `python -m src.preprocessing` from LTF_code/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils.config import load_config
from utils.logger import get_logger


# ============================================================================
# Path resolution helpers
# ============================================================================

def _resolve_data_path(cfg: SimpleNamespace) -> str:
    """Return absolute path to the dataset directory."""
    dataset_dir = cfg.data.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.normpath(os.path.join(_PROJECT_DIR, dataset_dir))
    return dataset_dir


# ============================================================================
# Step 1: File existence check
# ============================================================================

def verify_files(cfg: SimpleNamespace, logger) -> Dict[str, str]:
    """Check that all required data files exist.

    Returns:
        Dict mapping logical names to absolute file paths.

    Raises:
        FileNotFoundError if any required file is missing.
    """
    dataset_dir = _resolve_data_path(cfg)
    logger.info(f"Dataset directory: {dataset_dir}")

    required = {
        "content_features": os.path.join(dataset_dir, cfg.data.content_features_file),
        "temporal_target": os.path.join(dataset_dir, cfg.data.temporal_target_file),
        "metadata": os.path.join(dataset_dir, cfg.data.metadata_file),
    }

    paths = {}
    for name, path in required.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file missing: {name} -> {path}")
        paths[name] = path
        logger.info(f"  [OK] {name}: {path}")

    # Optional: user_info
    if cfg.data.user_info_file:
        ui_path = os.path.join(dataset_dir, cfg.data.user_info_file)
        if os.path.isfile(ui_path):
            paths["user_info"] = ui_path
            logger.info(f"  [OK] user_info (optional): {ui_path}")
        else:
            logger.warning(f"  [SKIP] user_info not found: {ui_path} — will use content_features social slice only")
    else:
        logger.info("  [SKIP] user_info_file is null in config")

    return paths


# ============================================================================
# Step 2: Content features dimension verification
# ============================================================================

def verify_content_features(
    path: str, cfg: SimpleNamespace, logger
) -> Dict[str, int]:
    """Load content_features.npy header and verify dimensions.

    Args:
        path:   Absolute path to content_features.npy.
        cfg:    Config namespace (uses cfg.feature_slicing).
        logger: Logger instance.

    Returns:
        Dict with verified slice boundaries:
        {
            "n_samples": int,
            "total_dim": int,
            "visual": (start, end),
            "acoustic": (start, end),
            "textual": (start, end),
            "metadata": (start, end),
            "social": (start, end),
        }

    Raises:
        ValueError if total dimension does not match expected slicing.
    """
    logger.info("--- Verifying content_features.npy ---")

    # Load with mmap to avoid reading 3.2 GB into RAM during verification
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    n_samples, total_dim = data.shape[0], data.shape[1]
    logger.info(f"  Shape: ({n_samples}, {total_dim}), dtype: {data.dtype}")

    fs = cfg.feature_slicing
    expected_known = fs.visual_dim + fs.acoustic_dim + fs.textual_dim + fs.metadata_dim
    social_dim = total_dim - expected_known

    if social_dim < 0:
        raise ValueError(
            f"Content features total dim ({total_dim}) is smaller than "
            f"sum of known slices ({expected_known}). "
            f"Check feature_slicing in config."
        )

    # Build slice boundaries
    idx = 0
    slices = {}
    for name, dim in [
        ("visual", fs.visual_dim),
        ("acoustic", fs.acoustic_dim),
        ("textual", fs.textual_dim),
        ("metadata", fs.metadata_dim),
    ]:
        slices[name] = (idx, idx + dim)
        idx += dim
    slices["social"] = (idx, idx + social_dim)

    logger.info(f"  visual:   [{slices['visual'][0]}:{slices['visual'][1]}]  dim={fs.visual_dim}")
    logger.info(f"  acoustic: [{slices['acoustic'][0]}:{slices['acoustic'][1]}]  dim={fs.acoustic_dim}")
    logger.info(f"  textual:  [{slices['textual'][0]}:{slices['textual'][1]}]  dim={fs.textual_dim}")
    logger.info(f"  metadata: [{slices['metadata'][0]}:{slices['metadata'][1]}]  dim={fs.metadata_dim}")
    logger.info(f"  social:   [{slices['social'][0]}:{slices['social'][1]}]  dim={social_dim} (inferred)")

    return {
        "n_samples": n_samples,
        "total_dim": total_dim,
        "social_dim": social_dim,
        **slices,
    }


# ============================================================================
# Step 3: Temporal data + label verification
# ============================================================================

def verify_temporal_target(
    path: str, cfg: SimpleNamespace, logger
) -> Dict[str, int]:
    """Load temporal_data_target.npy and verify layout.

    Expected per-sample layout (from TIML code):
        [short_term: 72*4=288] [long_term: 6*3=18] [t_p: 3] [label: 1]
        Total = 310

    Args:
        path:   Absolute path to temporal_data_target.npy.
        cfg:    Config namespace (uses cfg.temporal_layout).
        logger: Logger instance.

    Returns:
        Dict with verified layout info and label statistics.

    Raises:
        ValueError if total dimension does not match expected layout.
    """
    logger.info("--- Verifying temporal_data_target.npy ---")

    data = np.load(path, allow_pickle=True, mmap_mode="r")
    n_samples = data.shape[0]
    total_dim = data.shape[1] if data.ndim == 2 else None
    logger.info(f"  Shape: {data.shape}, dtype: {data.dtype}")

    tl = cfg.temporal_layout
    short_total = tl.short_steps * tl.short_features   # 72*4 = 288
    long_total = tl.long_steps * tl.long_features       # 6*3  = 18
    expected_total = short_total + long_total + tl.t_p_dim + tl.label_dim  # 310

    if total_dim != expected_total:
        raise ValueError(
            f"temporal_data_target per-sample dim is {total_dim}, "
            f"expected {expected_total} "
            f"(short={short_total} + long={long_total} + t_p={tl.t_p_dim} + label={tl.label_dim})"
        )

    logger.info(f"  short_term: [0:{short_total}]  ({tl.short_steps} steps x {tl.short_features} feats)")
    logger.info(f"  long_term:  [{short_total}:{short_total + long_total}]  ({tl.long_steps} steps x {tl.long_features} feats)")
    logger.info(f"  t_p:        [{short_total + long_total}:{short_total + long_total + tl.t_p_dim}]")
    logger.info(f"  label:      [{expected_total - 1}]")

    # Label statistics — read only the label column
    labels = np.array(data[:, -1], dtype=np.float32)
    label_stats = {
        "count": int(n_samples),
        "min": float(np.min(labels)),
        "max": float(np.max(labels)),
        "mean": float(np.mean(labels)),
        "std": float(np.std(labels)),
        "median": float(np.median(labels)),
    }
    logger.info(f"  Label stats: min={label_stats['min']:.4f}, max={label_stats['max']:.4f}, "
                f"mean={label_stats['mean']:.4f}, std={label_stats['std']:.4f}, "
                f"median={label_stats['median']:.4f}")

    # Check if label looks log-scaled (positive, moderate range)
    if label_stats["min"] >= 0 and label_stats["max"] < 30:
        logger.info("  Label appears log-scaled (range [0, ~30)), consistent with log(popularity+1)")
    else:
        logger.warning(f"  Label range [{label_stats['min']}, {label_stats['max']}] — verify if log-scaled")

    return {
        "n_samples": n_samples,
        "total_dim": total_dim,
        "short_total": short_total,
        "long_total": long_total,
        "label_stats": label_stats,
    }


# ============================================================================
# Step 4: Sample count alignment
# ============================================================================

def verify_alignment(content_info: Dict, temporal_info: Dict, logger) -> int:
    """Verify that content_features and temporal_data_target have the same sample count.

    Returns:
        Number of aligned samples.

    Raises:
        ValueError if sample counts differ.
    """
    n_content = content_info["n_samples"]
    n_temporal = temporal_info["n_samples"]

    if n_content != n_temporal:
        raise ValueError(
            f"Sample count mismatch: content_features has {n_content}, "
            f"temporal_data_target has {n_temporal}"
        )

    logger.info(f"--- Sample alignment OK: {n_content} samples ---")
    return n_content


# ============================================================================
# Step 5: Generate reproducible train/val/test split indices
# ============================================================================

def generate_splits(
    n_samples: int, cfg: SimpleNamespace, logger
) -> Dict[str, np.ndarray]:
    """Generate and save train/val/test index arrays.

    Uses a seeded permutation so splits are deterministic.

    Args:
        n_samples: Total number of samples.
        cfg:       Config namespace (uses cfg.training).
        logger:    Logger instance.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to an int array of indices.
    """
    seed = cfg.training.seed
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)

    train_r = cfg.training.train_ratio
    val_r = cfg.training.val_ratio
    # test_ratio is implicit: 1 - train - val

    n_train = int(n_samples * train_r)
    n_val = int(n_samples * val_r)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    logger.info(f"--- Split sizes (seed={seed}): "
                f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])} ---")

    # Save to processed dir
    processed_dir = cfg.data.processed_dir
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(_PROJECT_DIR, processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    split_path = os.path.join(processed_dir, f"splits_seed{seed}.npz")
    np.savez(split_path, train=splits["train"], val=splits["val"], test=splits["test"])
    logger.info(f"  Saved splits to: {split_path}")

    return splits


# ============================================================================
# Step 6: Save verification report
# ============================================================================

def save_report(
    content_info: Dict,
    temporal_info: Dict,
    n_samples: int,
    cfg: SimpleNamespace,
    logger,
) -> str:
    """Save a JSON verification report to the processed directory.

    Returns:
        Path to the saved report file.
    """
    processed_dir = cfg.data.processed_dir
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(_PROJECT_DIR, processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    report = {
        "n_samples": n_samples,
        "content_features": {
            "total_dim": content_info["total_dim"],
            "visual": list(content_info["visual"]),
            "acoustic": list(content_info["acoustic"]),
            "textual": list(content_info["textual"]),
            "metadata": list(content_info["metadata"]),
            "social": list(content_info["social"]),
            "social_dim": content_info["social_dim"],
        },
        "temporal_target": {
            "total_dim": temporal_info["total_dim"],
            "short_total": temporal_info["short_total"],
            "long_total": temporal_info["long_total"],
            "label_stats": temporal_info["label_stats"],
        },
    }

    report_path = os.path.join(processed_dir, "data_verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"--- Verification report saved to: {report_path} ---")
    return report_path


# ============================================================================
# Main orchestrator
# ============================================================================

def run_preprocessing(config_path: str) -> Dict:
    """Run the full preprocessing pipeline.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dict containing content_info, temporal_info, n_samples, splits.
    """
    cfg = load_config(config_path)
    logger = get_logger("preprocessing", log_dir=cfg.output.log_dir)

    logger.info("=" * 60)
    logger.info("XMTL Data Preprocessing & Verification")
    logger.info("=" * 60)

    # Step 1: File existence
    paths = verify_files(cfg, logger)

    # Step 2: Content features dimensions
    content_info = verify_content_features(paths["content_features"], cfg, logger)

    # Step 3: Temporal + label
    temporal_info = verify_temporal_target(paths["temporal_target"], cfg, logger)

    # Step 4: Alignment
    n_samples = verify_alignment(content_info, temporal_info, logger)

    # Step 5: Splits
    splits = generate_splits(n_samples, cfg, logger)

    # Step 6: Report
    save_report(content_info, temporal_info, n_samples, cfg, logger)

    logger.info("=" * 60)
    logger.info("Preprocessing complete. All checks passed.")
    logger.info("=" * 60)

    return {
        "content_info": content_info,
        "temporal_info": temporal_info,
        "n_samples": n_samples,
        "splits": splits,
        "paths": paths,
    }


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XMTL Data Preprocessing")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run_preprocessing(args.config)
