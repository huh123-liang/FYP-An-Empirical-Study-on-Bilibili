"""
src/dataset.py
--------------
XMTLDataset: PyTorch Dataset for the XMTL framework.

Loads content_features.npy and temporal_data_target.npy, slices them into
per-modality tensors according to the verified config, and supports
modality masking for ablation studies.

Data arrays can be shared across train/val/test splits to avoid loading
the same 1.6GB file multiple times. Use load_shared_data() + pass
shared_data to the constructor.

Usage:
    from src.dataset import XMTLDataset, load_shared_data
    shared = load_shared_data(cfg)
    ds = XMTLDataset(cfg, shared_data=shared)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def load_shared_data(cfg: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray]:
    """Load data arrays once, to be shared across dataset instances.

    Returns:
        (content_features [N, 8353], temporal_target [N, 310]) as float32.
    """
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


class XMTLDataset(Dataset):
    """XMTL multi-modal temporal dataset.

    Args:
        cfg:            Config namespace from load_config().
        indices:        Optional array of sample indices (for train/val/test split).
        modality_mask:  Optional dict like {"visual": False} to zero-mask modalities.
        shared_data:    Optional (content, temporal) tuple from load_shared_data().
        content_stats:  Optional (mean, std) arrays for content feature standardization.
    """

    def __init__(
        self,
        cfg: SimpleNamespace,
        indices: Optional[np.ndarray] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
        shared_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        content_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        super().__init__()
        if shared_data is not None:
            self.content_features, self.temporal_target = shared_data
        else:
            self.content_features, self.temporal_target = load_shared_data(cfg)

        if indices is not None:
            self.indices = np.sort(indices)
        else:
            self.indices = np.arange(self.content_features.shape[0])

        # Global standardization stats (computed from training set)
        self.content_mean = content_stats[0] if content_stats else None
        self.content_std = content_stats[1] if content_stats else None

        # Content feature slice boundaries
        fs = cfg.feature_slicing
        v, a, t, m = fs.visual_dim, fs.acoustic_dim, fs.textual_dim, fs.metadata_dim
        s = self.content_features.shape[1] - (v + a + t + m)
        self.content_slices = {
            "visual":   (0, v),
            "acoustic": (v, v + a),
            "textual":  (v + a, v + a + t),
            "metadata": (v + a + t, v + a + t + m),
            "creator":  (v + a + t + m, v + a + t + m + s),
        }

        # Temporal layout
        tl = cfg.temporal_layout
        self.short_total = tl.short_steps * tl.short_features
        self.long_total = tl.long_steps * tl.long_features
        self.short_steps = tl.short_steps
        self.short_feats = tl.short_features
        self.long_steps = tl.long_steps
        self.long_feats = tl.long_features
        self.t_p_dim = tl.t_p_dim

        default_mask = {
            "visual": True, "acoustic": True, "textual": True,
            "metadata": True, "creator": True, "temporal": True,
        }
        if modality_mask:
            default_mask.update(modality_mask)
        self.modality_mask = default_mask

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample as a dict of tensors.

        Returns:
            visual [4096], acoustic [2688], textual [1538], metadata [22],
            creator [9], temporal_short [72,4], temporal_long [6,3],
            t_p [3], label [1]
        """
        real_idx = self.indices[idx]
        content_row = self.content_features[real_idx].copy()
        temporal_row = self.temporal_target[real_idx]

        # Apply global standardization if stats are available
        if self.content_mean is not None:
            content_row = (content_row - self.content_mean) / self.content_std

        sample = {}
        for name, (start, end) in self.content_slices.items():
            feat = torch.from_numpy(content_row[start:end].copy())
            if not self.modality_mask.get(name, True):
                feat = torch.zeros_like(feat)
            sample[name] = feat

        st, lt = self.short_total, self.long_total
        short_raw = temporal_row[:st]
        long_raw = temporal_row[st:st + lt]
        t_p = temporal_row[st + lt:st + lt + self.t_p_dim]
        label = temporal_row[-1:]

        temporal_short = torch.from_numpy(
            short_raw.reshape(self.short_steps, self.short_feats).copy()
        )
        temporal_long = torch.from_numpy(
            long_raw.reshape(self.long_steps, self.long_feats).copy()
        )

        if not self.modality_mask.get("temporal", True):
            temporal_short = torch.zeros_like(temporal_short)
            temporal_long = torch.zeros_like(temporal_long)
            t_p = np.zeros_like(t_p)

        sample["temporal_short"] = temporal_short
        sample["temporal_long"] = temporal_long
        sample["t_p"] = torch.from_numpy(t_p.copy())
        sample["label"] = torch.from_numpy(label.copy())
        return sample
