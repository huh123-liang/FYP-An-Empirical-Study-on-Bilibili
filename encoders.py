"""
models/encoders.py
------------------
Modality-specific encoders for the XMTL framework.

Content encoders (2-layer MLP each):
    VisualEncoder:    [B, 4096] -> [B, 64]
    AcousticEncoder:  [B, 2688] -> [B, 64]
    TextualEncoder:   [B, 1538] -> [B, 64]
    MetadataEncoder:  [B, 22]   -> [B, 64]
    CreatorEncoder:   [B, 9]    -> [B, 64]

Temporal encoder (hierarchical GRU with residual injection + triple pooling):
    TemporalEncoder:
        short: [B, 72, 4], long: [B, 6, 3], t_p: [B, 3] -> [B, 64]
"""

import torch
import torch.nn as nn
from typing import Tuple


# ============================================================================
# Content Encoders — 2-layer MLP: input -> hidden -> ReLU -> Dropout -> out
# ============================================================================

class ContentEncoder(nn.Module):
    """Generic 2-layer MLP encoder for pre-extracted content features.

    Args:
        input_dim:  Raw feature dimension.
        hidden_dim: Intermediate dimension.
        output_dim: Encoder output dimension (default 256).
        dropout:    Dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),          # [B, input_dim] -> [B, hidden_dim]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),         # [B, hidden_dim] -> [B, output_dim]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            [B, output_dim]
        """
        return self.net(x)


class VisualEncoder(ContentEncoder):
    """[B, 4096] -> [B, 128]"""
    def __init__(self, output_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim=4096, hidden_dim=256, output_dim=output_dim,
                         dropout=dropout)


class AcousticEncoder(ContentEncoder):
    """[B, 2688] -> [B, 128]"""
    def __init__(self, output_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim=2688, hidden_dim=256, output_dim=output_dim,
                         dropout=dropout)


class TextualEncoder(ContentEncoder):
    """[B, 1538] -> [B, 128]"""
    def __init__(self, output_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim=1538, hidden_dim=128, output_dim=output_dim,
                         dropout=dropout)


class MetadataEncoder(ContentEncoder):
    """[B, 22] -> [B, 128]"""
    def __init__(self, output_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim=22, hidden_dim=64, output_dim=output_dim,
                         dropout=dropout)


class CreatorEncoder(ContentEncoder):
    """[B, 9] -> [B, 128]"""
    def __init__(self, output_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim=9, hidden_dim=64, output_dim=output_dim,
                         dropout=dropout)


# ============================================================================
# Temporal Encoder — Hierarchical GRU with residual injection + triple pooling
# ============================================================================

class TemporalEncoder(nn.Module):
    """Hierarchical temporal encoder.

    Architecture:
        1. short_gru: processes short-term sequence [B, 72, 4]
           - Triple pooling: last_hidden + mean(all_hidden) + max(all_hidden)
           - Project: Linear(128*3, 64) -> short_summary [B, 64]

        2. Residual injection: short_summary is repeated across long-term steps
           and concatenated with long-term features as context.
           long_input = [long_features; short_summary.repeat(T_long)]  -> [B, 6, 3+64]

        3. long_gru: processes enriched long-term sequence [B, 6, 67]
           - Uses last hidden state [B, 64]

        4. Final projection: Linear(64 + 3, 64) merging long_gru output with t_p

    Args:
        short_input_dim: Feature dim per short-term step (default 4).
        long_input_dim:  Feature dim per long-term step (default 3).
        t_p_dim:         Temporal context dimension (default 3).
        short_hidden:    Short GRU hidden size (default 128).
        long_hidden:     Long GRU hidden size (default 64).
        output_dim:      Final output dimension (default 64).
        dropout:         Dropout probability.
    """

    def __init__(
        self,
        short_input_dim: int = 4,
        long_input_dim: int = 3,
        t_p_dim: int = 3,
        short_hidden: int = 128,
        long_hidden: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.short_hidden = short_hidden
        self.long_hidden = long_hidden

        # Input normalization for temporal sequences
        self.short_norm = nn.LayerNorm(short_input_dim)
        self.long_norm = nn.LayerNorm(long_input_dim)

        # Short-term GRU
        self.short_gru = nn.GRU(
            input_size=short_input_dim,
            hidden_size=short_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Triple pooling projection: (last + avg + max) each [B, short_hidden]
        # -> [B, short_hidden * 3] -> [B, output_dim]
        self.short_proj = nn.Sequential(
            nn.Linear(short_hidden * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Long-term GRU with residual injection from short summary
        # input_size = long_input_dim + output_dim (residual context)
        self.long_gru = nn.GRU(
            input_size=long_input_dim + output_dim,
            hidden_size=long_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Final projection: long_gru output + t_p -> output_dim
        self.final_proj = nn.Sequential(
            nn.Linear(long_hidden + t_p_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        temporal_short: torch.Tensor,
        temporal_long: torch.Tensor,
        t_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            temporal_short: [B, 72, 4]  short-term hourly sequence
            temporal_long:  [B, 6, 3]   long-term daily sequence
            t_p:            [B, 3]      temporal context (holiday, pub_day_count)

        Returns:
            temporal_repr:  [B, output_dim]
        """
        B = temporal_short.size(0)

        # --- Normalize inputs ---
        temporal_short = self.short_norm(temporal_short)  # [B, 72, 4]
        temporal_long = self.long_norm(temporal_long)     # [B, 6, 3]

        # --- Short-term branch ---
        # short_out: [B, 72, short_hidden], short_hn: [1, B, short_hidden]
        short_out, short_hn = self.short_gru(temporal_short)

        # Triple pooling
        short_last = short_hn.squeeze(0)                    # [B, 128]
        short_avg = short_out.mean(dim=1)                   # [B, 128]
        short_max = short_out.max(dim=1).values             # [B, 128]
        short_pooled = torch.cat([short_last, short_avg, short_max], dim=1)  # [B, 384]

        short_summary = self.short_proj(short_pooled)       # [B, 64]

        # --- Residual injection into long-term branch ---
        T_long = temporal_long.size(1)  # 6
        # Repeat short_summary across time steps: [B, 64] -> [B, 6, 64]
        short_ctx = short_summary.unsqueeze(1).expand(-1, T_long, -1)
        # Concatenate with long-term features: [B, 6, 3+64] = [B, 6, 67]
        long_input = torch.cat([temporal_long, short_ctx], dim=2)

        # --- Long-term branch ---
        # long_out: [B, 6, long_hidden], long_hn: [1, B, long_hidden]
        _, long_hn = self.long_gru(long_input)
        long_repr = long_hn.squeeze(0)                      # [B, 64]

        # --- Merge with temporal context ---
        merged = torch.cat([long_repr, t_p], dim=1)         # [B, 64+3]
        temporal_repr = self.final_proj(merged)              # [B, 64]

        return temporal_repr
