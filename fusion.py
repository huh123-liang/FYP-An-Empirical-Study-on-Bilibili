"""
models/fusion.py
----------------
Multimodal fusion modules for XMTL.

All fusion modules accept:
    modality_reprs: list of [B, D] tensors (one per active modality)
    t_p:            [B, 3] temporal context

All fusion modules return:
    fused:           [B, D]  fused representation
    modality_weights: [B, M] normalized weights (M = number of modalities)

Available fusion types (selected via config.model.fusion_type):
    - GatedFusion:     temporal-aware gated fusion with temperature
    - AttentionFusion: scaled dot-product attention with temporal query
    - ConcatFusion:    concatenate + project (no explicit weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class GatedFusion(nn.Module):
    """Temporal-aware gated fusion with temperature scaling.

    For each modality i:
        g_i = sigma( W_gate @ [h_i; t_ctx] + b )
    Normalized:
        w = softmax([g_1, ..., g_M] / tau)
    Fused:
        h = sum(w_i * h_i)

    Args:
        embed_dim:  Dimension of each modality representation (default 64).
        t_p_dim:    Dimension of temporal context (default 3).
        n_modalities: Maximum number of modalities (default 6).
        temperature: Softmax temperature (default 0.5). Lower = sharper.
        dropout:    Dropout on temporal context projection.
    """

    def __init__(self, embed_dim: int = 64, t_p_dim: int = 3,
                 n_modalities: int = 6, temperature: float = 0.5,
                 dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature

        # Project temporal context to embed_dim
        self.t_proj = nn.Sequential(
            nn.Linear(t_p_dim, embed_dim),    # [B, 3] -> [B, 64]
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gate network: [h_i; t_ctx] -> scalar gate
        self.gate = nn.Linear(embed_dim + embed_dim, 1)  # [B, 128] -> [B, 1]

    def forward(
        self,
        modality_reprs: List[torch.Tensor],
        t_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_reprs: List of M tensors, each [B, D].
            t_p:            [B, 3] temporal context.

        Returns:
            fused:            [B, D]
            modality_weights: [B, M]
        """
        t_ctx = self.t_proj(t_p)  # [B, D]

        gates = []
        for h in modality_reprs:
            # [B, D] cat [B, D] -> [B, 2D] -> [B, 1]
            g = self.gate(torch.cat([h, t_ctx], dim=1))
            gates.append(g)

        # Stack gates: [B, M]
        gates = torch.cat(gates, dim=1)
        # Temperature-scaled softmax
        weights = F.softmax(gates / self.temperature, dim=1)  # [B, M]

        # Weighted sum
        stacked = torch.stack(modality_reprs, dim=1)  # [B, M, D]
        fused = (weights.unsqueeze(2) * stacked).sum(dim=1)  # [B, D]

        return fused, weights


class AttentionFusion(nn.Module):
    """Scaled dot-product attention fusion with temporal query.

    Query = temporal context projection.
    Keys = Values = modality representations.

    Args:
        embed_dim:  Dimension of each modality representation.
        t_p_dim:    Dimension of temporal context.
        temperature: Attention temperature.
        dropout:    Attention dropout.
    """

    def __init__(self, embed_dim: int = 64, t_p_dim: int = 3,
                 temperature: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.q_proj = nn.Linear(t_p_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality_reprs: List[torch.Tensor],
        t_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_reprs: List of M tensors, each [B, D].
            t_p:            [B, 3].

        Returns:
            fused:            [B, D]
            modality_weights: [B, M]
        """
        query = self.q_proj(t_p)  # [B, D]
        stacked = torch.stack(modality_reprs, dim=1)  # [B, M, D]
        keys = self.k_proj(stacked)  # [B, M, D]

        # Scaled dot-product: [B, M]
        scores = (query.unsqueeze(1) * keys).sum(dim=2)  # [B, M]
        weights = F.softmax(scores / self.temperature, dim=1)
        weights = self.dropout(weights)

        fused = (weights.unsqueeze(2) * stacked).sum(dim=1)  # [B, D]
        return fused, weights


class ConcatFusion(nn.Module):
    """Concatenate all modality representations and project.

    Returns uniform weights (1/M) since there is no explicit gating.

    Args:
        embed_dim:    Dimension of each modality representation.
        n_modalities: Number of modalities.
        dropout:      Dropout on projection.
    """

    def __init__(self, embed_dim: int = 64, n_modalities: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.n_modalities = n_modalities
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        modality_reprs: List[torch.Tensor],
        t_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_reprs: List of M tensors, each [B, D].
            t_p:            [B, 3] (unused, kept for interface consistency).

        Returns:
            fused:            [B, D]
            modality_weights: [B, M] uniform weights.
        """
        B = modality_reprs[0].size(0)
        M = len(modality_reprs)

        concatenated = torch.cat(modality_reprs, dim=1)  # [B, M*D]
        fused = self.proj(concatenated)  # [B, D]

        # Uniform weights
        weights = torch.full((B, M), 1.0 / M,
                             device=fused.device, dtype=fused.dtype)
        return fused, weights
