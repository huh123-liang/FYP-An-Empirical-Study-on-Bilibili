"""
models/xmtl.py
--------------
XMTL: Explainable Multimodal Temporal Learning model.

Assembles modality encoders, fusion module, and prediction head into a
single nn.Module. Supports temporal-only mode (use_content=False) for
ablation studies.

Forward returns a dict:
    prediction:       [B, 1]
    modality_weights: [B, M]  (M = number of active modalities)
    fused_repr:       [B, D]

Usage:
    from models.xmtl import build_xmtl
    model = build_xmtl(cfg)
    out = model(batch)
"""

import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Dict

from models.encoders import (
    VisualEncoder, AcousticEncoder, TextualEncoder,
    MetadataEncoder, CreatorEncoder, TemporalEncoder,
)
from models.fusion import GatedFusion, AttentionFusion, ConcatFusion


class XMTL(nn.Module):
    """Explainable Multimodal Temporal Learning model.

    Args:
        encoder_dim:  Output dimension of each encoder (default 128).
        fusion_type:  'gated' | 'attention' | 'concat'.
        temperature:  Softmax temperature for fusion gates.
        dropout:      Dropout probability.
        use_content:  If False, only temporal branch is used (for ablation).
        t_p_dim:      Temporal context dimension (default 3).
    """
    def __init__(
        self,
        encoder_dim: int = 128,
        fusion_type: str = "gated",
        temperature: float = 0.5,
        dropout: float = 0.1,
        use_content: bool = True,
        t_p_dim: int = 3,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.use_content = use_content

        # --- Content encoders (only built if use_content=True) ---
        if use_content:
            self.visual_enc = VisualEncoder(output_dim=encoder_dim, dropout=dropout)
            self.acoustic_enc = AcousticEncoder(output_dim=encoder_dim, dropout=dropout)
            self.textual_enc = TextualEncoder(output_dim=encoder_dim, dropout=dropout)
            self.metadata_enc = MetadataEncoder(output_dim=encoder_dim, dropout=dropout)
            self.creator_enc = CreatorEncoder(output_dim=encoder_dim, dropout=dropout)

        # --- Temporal encoder (always active) ---
        self.temporal_enc = TemporalEncoder(
            short_input_dim=4, long_input_dim=3, t_p_dim=t_p_dim,
            short_hidden=128, long_hidden=64,
            output_dim=encoder_dim, dropout=dropout,
        )

        # --- Fusion ---
        n_modalities = 6 if use_content else 1
        if n_modalities == 1:
            # Temporal-only: no fusion needed, pass through
            self.fusion = None
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                embed_dim=encoder_dim, t_p_dim=t_p_dim,
                n_modalities=n_modalities, temperature=temperature,
                dropout=dropout,
            )
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(
                embed_dim=encoder_dim, t_p_dim=t_p_dim,
                temperature=temperature, dropout=dropout,
            )
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(
                embed_dim=encoder_dim, n_modalities=n_modalities,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # --- 2-layer prediction head ---
        self.pred_head = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, orthogonal for GRU."""
        for name, param in self.named_parameters():
            if "gru" in name and "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dict from XMTLDataset.__getitem__ with keys:
                visual [B, 4096], acoustic [B, 2688], textual [B, 1538],
                metadata [B, 22], creator [B, 9],
                temporal_short [B, 72, 4], temporal_long [B, 6, 3],
                t_p [B, 3], label [B, 1]

        Returns:
            Dict with keys:
                prediction:       [B, 1]
                modality_weights: [B, M]
                fused_repr:       [B, D]
        """
        t_p = batch["t_p"]  # [B, 3]

        # --- Temporal (always) ---
        temporal_repr = self.temporal_enc(
            batch["temporal_short"], batch["temporal_long"], t_p
        )  # [B, 64]

        if self.use_content:
            # --- Content encoders ---
            visual_repr = self.visual_enc(batch["visual"])       # [B, 64]
            acoustic_repr = self.acoustic_enc(batch["acoustic"]) # [B, 64]
            textual_repr = self.textual_enc(batch["textual"])    # [B, 64]
            metadata_repr = self.metadata_enc(batch["metadata"]) # [B, 64]
            creator_repr = self.creator_enc(batch["creator"])    # [B, 64]

            # Order: visual, acoustic, textual, metadata, creator, temporal
            modality_reprs = [
                visual_repr, acoustic_repr, textual_repr,
                metadata_repr, creator_repr, temporal_repr,
            ]

            # --- Fusion ---
            fused, weights = self.fusion(modality_reprs, t_p)
        else:
            # Temporal-only mode
            fused = temporal_repr
            weights = torch.ones(
                temporal_repr.size(0), 1,
                device=temporal_repr.device, dtype=temporal_repr.dtype,
            )

        # --- Prediction ---
        prediction = self.pred_head(fused)  # [B, 1]

        return {
            "prediction": prediction,
            "modality_weights": weights,
            "fused_repr": fused,
        }


# ============================================================================
# Factory
# ============================================================================

def build_xmtl(cfg: SimpleNamespace) -> XMTL:
    """Build XMTL model from config.

    Args:
        cfg: Config namespace with cfg.model.* fields.

    Returns:
        XMTL model instance.
    """
    use_content = getattr(cfg.model, "use_content", True)
    temperature = getattr(cfg.model, "temperature", 0.5)

    return XMTL(
        encoder_dim=cfg.model.encoder_dim,
        fusion_type=cfg.model.fusion_type,
        temperature=temperature,
        dropout=cfg.model.dropout,
        use_content=use_content,
        t_p_dim=cfg.temporal_layout.t_p_dim,
    )
