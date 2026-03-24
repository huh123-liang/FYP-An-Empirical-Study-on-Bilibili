"""
training/train.py
-----------------
CLI entry point for training the XMTL model.

Loads config, builds dataloaders and model, trains with early stopping,
evaluates on test set, and saves all results.

Usage:
    cd LTF_code/
    python -m training.train --config configs/default.yaml
    python -m training.train --config configs/default.yaml --temporal_only
"""

import os
import sys
import argparse
import torch
from types import SimpleNamespace

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import get_logger
from src.dataloader import build_dataloaders
from models.xmtl import build_xmtl
from training.trainer import Trainer


def resolve_device(cfg: SimpleNamespace) -> torch.device:
    """Resolve device from config."""
    if cfg.training.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.training.device)


def main(config_path: str, temporal_only: bool = False):
    cfg = load_config(config_path)
    set_seed(cfg.training.seed)

    # Override use_content if temporal_only flag is set
    if temporal_only:
        cfg.model.use_content = False

    device = resolve_device(cfg)
    mode = "temporal_only" if not getattr(cfg.model, "use_content", True) else "full"

    logger = get_logger(f"xmtl_{mode}", log_dir=cfg.output.log_dir)
    logger.info("=" * 60)
    logger.info(f"XMTL Training — mode: {mode}, device: {device}")
    logger.info("=" * 60)

    # --- Dataloaders ---
    logger.info("Building dataloaders...")
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
    logger.info(f"  Train: {len(train_dl.dataset)} samples, {len(train_dl)} batches")
    logger.info(f"  Val:   {len(val_dl.dataset)} samples")
    logger.info(f"  Test:  {len(test_dl.dataset)} samples")

    # --- Model ---
    model = build_xmtl(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {n_params:,}")

    # --- Save directory ---
    result_dir = cfg.output.result_dir
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(_PROJECT_DIR, result_dir)
    save_dir = os.path.join(result_dir, f"xmtl_{mode}")
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    import json
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2, default=str)

    # --- Train ---
    trainer = Trainer(model, cfg, logger, device, save_dir)
    trainer.fit(train_dl, val_dl)

    # --- Test ---
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test(test_dl)

    logger.info("=" * 60)
    logger.info(f"XMTL {mode} complete.")
    logger.info(f"  MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}  "
                f"Spearman={test_metrics['spearman']:.4f}  R2={test_metrics['r2']:.4f}")
    logger.info(f"  Results saved to: {save_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XMTL Model Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--temporal_only", action="store_true",
        help="Train temporal-only model (no content encoders)",
    )
    args = parser.parse_args()
    main(args.config, temporal_only=args.temporal_only)
