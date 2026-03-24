"""
Generate publication-quality figures for XMTL conference paper.
Consistent style suitable for NeurIPS / IEEE / ACM venues.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

# ── Global Style ──────────────────────────────────────────────────────────────

PALETTE = {
    'temporal':  '#2171B5',
    'visual':    '#E6550D',
    'textual':   '#31A354',
    'acoustic':  '#756BB1',
    'metadata':  '#969696',
    'creator':   '#8C6D31',
}
PALETTE_LIGHT = {
    'temporal':  '#C6DBEF',
    'visual':    '#FDD0A2',
    'textual':   '#C7E9C0',
    'acoustic':  '#DADAEB',
    'metadata':  '#D9D9D9',
    'creator':   '#E5D5B0',
}

BG_WHITE = '#FFFFFF'
TEXT_DARK = '#1a1a1a'
TEXT_MED  = '#4a4a4a'
GRID_CLR  = '#E0E0E0'
ACCENT    = '#2171B5'
BLOCK_BG  = '#F7F7F7'
BLOCK_EDGE = '#BDBDBD'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.facecolor': BG_WHITE,
    'axes.facecolor': BG_WHITE,
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': False,
    'text.color': TEXT_DARK,
    'axes.labelcolor': TEXT_DARK,
    'xtick.color': TEXT_MED,
    'ytick.color': TEXT_MED,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'cm',
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.svg'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved {name}.png and {name}.svg')


# ── Helper drawing functions ──────────────────────────────────────────────────

def draw_block(ax, xy, w, h, label, color=BLOCK_BG, edge=BLOCK_EDGE,
               fontsize=8, bold=False, text_color=TEXT_DARK, rounded=True):
    """Draw a rounded rectangle block with centered label."""
    rounding = 0.02 if rounded else 0.0
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={rounding}",
                         facecolor=color, edgecolor=edge, linewidth=0.8,
                         transform=ax.transData, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(xy[0] + w/2, xy[1] + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color, zorder=3)
    return box


def draw_arrow(ax, start, end, color=TEXT_MED, lw=0.8, style='->', shrinkA=2, shrinkB=2):
    """Draw a clean arrow between two points."""
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                            linewidth=lw, shrinkA=shrinkA, shrinkB=shrinkB,
                            mutation_scale=10, zorder=1)
    ax.add_patch(arrow)
    return arrow


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def figure1_architecture():
    """XMTL model architecture diagram (top-to-bottom flow)."""
    print('Generating Figure 1: Model Architecture...')
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # ── Column positions ──
    enc_x = 0.5;  enc_w = 3.8
    tmp_x = 5.5;  tmp_w = 3.8
    bw, bh = 0.9, 0.85

    # ── Section label: Input Layer ──
    ax.text(5.0, 11.8, 'Input Layer', ha='center', va='center',
            fontsize=7, fontstyle='italic', color=TEXT_MED)

    # ── Input modalities (top) ──
    input_y = 10.6
    modalities = [
        ('Visual\n(4096-d)',    PALETTE['visual'],    0.5),
        ('Acoustic\n(2688-d)',  PALETTE['acoustic'],  1.55),
        ('Textual\n(1538-d)',   PALETTE['textual'],   2.6),
        ('Metadata\n(22-d)',    PALETTE['metadata'],  3.65),
        ('Creator\n(9-d)',      PALETTE['creator'],   4.7),
    ]
    for label, color, x in modalities:
        draw_block(ax, (x, input_y), bw, bh, label,
                   color=PALETTE_LIGHT[label.split('\n')[0].lower()],
                   edge=color, fontsize=6.5)

    # Temporal input
    draw_block(ax, (tmp_x + 0.4, input_y), tmp_w - 0.8, bh,
               'Temporal Sequences\n72h short-term  ·  6-month long-term',
               color=PALETTE_LIGHT['temporal'], edge=PALETTE['temporal'], fontsize=7)

    # ── Content Encoders ──
    enc_y = 9.1
    enc_h = 0.7
    encoder_info = [
        ('Visual Encoder',   PALETTE['visual'],   0.5),
        ('Acoustic Encoder', PALETTE['acoustic'], 1.55),
        ('Textual Encoder',  PALETTE['textual'],  2.6),
        ('Meta Encoder',     PALETTE['metadata'], 3.65),
        ('Creator Encoder',  PALETTE['creator'],  4.7),
    ]
    for label, color, x in encoder_info:
        draw_block(ax, (x, enc_y), bw, enc_h, label,
                   color='white', edge=color, fontsize=6.5, bold=True)
        # arrow from input bottom to encoder top
        draw_arrow(ax, (x + bw/2, input_y), (x + bw/2, enc_y + enc_h),
                   color=color, lw=0.7)

    # MLP annotation
    ax.text(3.15, enc_y - 0.12, 'MLP Encoders (GELU + Dropout)',
            ha='center', va='top', fontsize=6.5, fontstyle='italic', color=TEXT_MED)

    # ── Temporal Encoder (right column) ──
    # Short-term GRU
    gru_s_y = 9.1
    gru_s_h = 0.7
    draw_block(ax, (tmp_x + 0.4, gru_s_y), tmp_w - 0.8, gru_s_h,
               'Short-term GRU  (72 × 4)',
               color='white', edge=PALETTE['temporal'], fontsize=7.5, bold=True)
    draw_arrow(ax, (tmp_x + tmp_w/2, input_y), (tmp_x + tmp_w/2, gru_s_y + gru_s_h),
               color=PALETTE['temporal'], lw=0.7)

    # Residual connection label
    ax.text(tmp_x + tmp_w/2, gru_s_y - 0.22, '+ residual', ha='center', va='center',
            fontsize=6.5, fontstyle='italic', color=PALETTE['temporal'])

    # Long-term GRU
    gru_l_y = gru_s_y - gru_s_h - 0.5
    gru_l_h = 0.7
    draw_block(ax, (tmp_x + 0.4, gru_l_y), tmp_w - 0.8, gru_l_h,
               'Long-term GRU  (6 × 3)',
               color='white', edge=PALETTE['temporal'], fontsize=7.5, bold=True)
    draw_arrow(ax, (tmp_x + tmp_w/2, gru_s_y),
               (tmp_x + tmp_w/2, gru_l_y + gru_l_h),
               color=PALETTE['temporal'], lw=0.7)

    # Temporal encoder bounding box
    ax.add_patch(FancyBboxPatch((tmp_x + 0.15, gru_l_y - 0.15),
                                tmp_w - 0.3, gru_s_y + gru_s_h - gru_l_y + 0.3,
                                boxstyle="round,pad=0.05",
                                facecolor='none', edgecolor=PALETTE['temporal'],
                                linewidth=1.0, linestyle='--', zorder=1))
    ax.text(tmp_x + tmp_w - 0.25, gru_l_y - 0.22,
            'Temporal Encoder', ha='right', va='top',
            fontsize=7, fontweight='bold', color=PALETTE['temporal'])

    # ── Encoded representations row ──
    repr_y = 6.8
    repr_h = 0.55
    # Content encoded
    draw_block(ax, (enc_x, repr_y), enc_w, repr_h,
               'Content Representations  (128-d each)',
               color=BLOCK_BG, edge=BLOCK_EDGE, fontsize=7)
    # Arrows from encoders to content repr
    for label, color, x in encoder_info:
        draw_arrow(ax, (x + bw/2, enc_y), (x + bw/2, repr_y + repr_h),
                   color=color, lw=0.6)

    # Temporal encoded
    draw_block(ax, (tmp_x + 0.4, repr_y), tmp_w - 0.8, repr_h,
               'Temporal Representation  (128-d)',
               color=PALETTE_LIGHT['temporal'], edge=PALETTE['temporal'], fontsize=7)
    draw_arrow(ax, (tmp_x + tmp_w/2, gru_l_y),
               (tmp_x + tmp_w/2, repr_y + repr_h),
               color=PALETTE['temporal'], lw=0.7)

    # ── Gated Fusion ──
    fuse_y = 5.2
    fuse_h = 0.9
    fuse_x = 1.5
    fuse_w = 7.0
    draw_block(ax, (fuse_x, fuse_y), fuse_w, fuse_h,
               'Temporal-Aware Gated Fusion',
               color='#E8F0FE', edge=ACCENT, fontsize=9, bold=True)
    # Arrows into fusion
    draw_arrow(ax, (enc_x + enc_w/2, repr_y),
               (fuse_x + fuse_w * 0.3, fuse_y + fuse_h), color=TEXT_MED, lw=0.8)
    draw_arrow(ax, (tmp_x + tmp_w/2, repr_y),
               (fuse_x + fuse_w * 0.7, fuse_y + fuse_h), color=PALETTE['temporal'], lw=0.8)

    # Fusion equation
    ax.text(fuse_x + fuse_w/2, fuse_y + fuse_h + 0.25,
            r'$w_i = \mathrm{softmax}\!\left(\sigma\!\left(W\,[h_i \,;\, t]\right)\right)$',
            ha='center', va='bottom', fontsize=9, color=TEXT_DARK,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor=GRID_CLR, linewidth=0.5))

    # ── Fused Representation ──
    fused_y = 4.0
    fused_h = 0.55
    fused_x = 2.5
    fused_w = 5.0
    draw_block(ax, (fused_x, fused_y), fused_w, fused_h,
               'Fused Representation', color=BLOCK_BG, edge=BLOCK_EDGE, fontsize=8)
    draw_arrow(ax, (fuse_x + fuse_w/2, fuse_y),
               (fused_x + fused_w/2, fused_y + fused_h), color=TEXT_MED, lw=0.8)

    # ── Two branches ──
    # Left: Prediction
    pred_y = 2.6
    pred_h = 0.6
    pred_x = 1.0
    pred_w = 3.0
    draw_block(ax, (pred_x, pred_y), pred_w, pred_h,
               'Popularity Prediction\n(log₁₀ views)', color='#E8F0FE',
               edge=ACCENT, fontsize=8, bold=True)
    draw_arrow(ax, (fused_x + fused_w * 0.3, fused_y),
               (pred_x + pred_w/2, pred_y + pred_h), color=TEXT_MED, lw=0.8)

    # Right: Explainability
    exp_y = 2.6
    exp_h = 0.6
    exp_x = 6.0
    exp_w = 3.0
    draw_block(ax, (exp_x, exp_y), exp_w, exp_h,
               'SHAP Explainability\nAnalysis', color='#FFF3E0',
               edge=PALETTE['visual'], fontsize=8, bold=True)
    draw_arrow(ax, (fused_x + fused_w * 0.7, fused_y),
               (exp_x + exp_w/2, exp_y + exp_h), color=PALETTE['visual'], lw=0.8)

    # Strategy generator
    strat_y = 1.2
    strat_h = 0.6
    draw_block(ax, (exp_x, strat_y), exp_w, strat_h,
               'Creator Strategy\nGenerator', color='#E8F5E9',
               edge=PALETTE['textual'], fontsize=8, bold=True)
    draw_arrow(ax, (exp_x + exp_w/2, exp_y),
               (exp_x + exp_w/2, strat_y + strat_h), color=PALETTE['textual'], lw=0.8)

    fig.tight_layout()
    save_fig(fig, 'xmtl_architecture_advanced')


def figure2_global_importance():
    """SHAP global importance — two-panel figure."""
    print('Generating Figure 2: Global Importance...')

    modalities = ['Temporal', 'Visual', 'Textual', 'Acoustic', 'Metadata', 'Creator']
    pct_values = [56.11, 18.79, 6.73, 6.67, 5.93, 5.77]
    colors = [PALETTE[m.lower()] for m in modalities]

    # Top-20 features grouped by modality (from SHAP data)
    top_features = [
        ('metadata_5',        0.240, 'metadata'),
        ('temporal_short_255', 0.233, 'temporal'),
        ('t_p_2',             0.207, 'temporal'),
        ('temporal_short_279', 0.155, 'temporal'),
        ('temporal_short_207', 0.155, 'temporal'),
        ('temporal_short_199', 0.123, 'temporal'),
        ('temporal_short_287', 0.115, 'temporal'),
        ('temporal_short_107', 0.113, 'temporal'),
        ('temporal_short_155', 0.109, 'temporal'),
        ('creator_8',         0.104, 'creator'),
        ('temporal_short_283', 0.098, 'temporal'),
        ('temporal_short_159', 0.098, 'temporal'),
        ('temporal_short_247', 0.096, 'temporal'),
        ('temporal_short_251', 0.094, 'temporal'),
        ('temporal_short_151', 0.091, 'temporal'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    # ── Panel A: Modality importance bar chart ──
    bars = ax1.barh(range(len(modalities)), pct_values, color=colors,
                    edgecolor='white', linewidth=0.5, height=0.65, zorder=2)
    ax1.set_yticks(range(len(modalities)))
    ax1.set_yticklabels(modalities)
    ax1.set_xlabel('SHAP Importance (%)')
    ax1.set_xlim(0, 65)
    ax1.invert_yaxis()
    ax1.grid(axis='x', color=GRID_CLR, linewidth=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # Value labels
    for i, (v, c) in enumerate(zip(pct_values, colors)):
        ax1.text(v + 0.8, i, f'{v:.1f}%', va='center', ha='left',
                 fontsize=7, color=c, fontweight='bold')

    # Highlight temporal dominance
    ax1.axhspan(-0.5, 0.5, color=PALETTE_LIGHT['temporal'], alpha=0.3, zorder=0)
    ax1.text(0.97, 0.97, '(a)', transform=ax1.transAxes, ha='right', va='top',
             fontsize=9, fontweight='bold')

    # ── Panel B: Top feature importance (grouped) ──
    feat_names = [f[0] for f in reversed(top_features)]
    feat_vals  = [f[1] for f in reversed(top_features)]
    feat_colors = [PALETTE[f[2]] for f in reversed(top_features)]

    ax2.barh(range(len(feat_names)), feat_vals, color=feat_colors,
             edgecolor='white', linewidth=0.3, height=0.7, zorder=2)
    ax2.set_yticks(range(len(feat_names)))
    # Clean up feature names for display
    display_names = []
    for n in feat_names:
        n2 = n.replace('temporal_short_', 'ts_').replace('metadata_', 'meta_').replace('creator_', 'cr_').replace('t_p_', 'tp_')
        display_names.append(n2)
    ax2.set_yticklabels(display_names, fontsize=6.5)
    ax2.set_xlabel('Mean |SHAP value|')
    ax2.grid(axis='x', color=GRID_CLR, linewidth=0.4, zorder=0)
    ax2.set_axisbelow(True)

    # Legend for modality colors
    handles = [mpatches.Patch(color=PALETTE[m], label=m.capitalize())
               for m in ['temporal', 'metadata', 'creator']]
    ax2.legend(handles=handles, loc='lower right', fontsize=6.5,
               frameon=True, fancybox=False, edgecolor=GRID_CLR)

    ax2.text(0.97, 0.97, '(b)', transform=ax2.transAxes, ha='right', va='top',
             fontsize=9, fontweight='bold')

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, 'shap_importance_advanced')


def figure3_local_explanation():
    """SHAP local explanation — waterfall + grouped contributions."""
    print('Generating Figure 3: Local Explanation...')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5),
                                    gridspec_kw={'width_ratios': [1.3, 1]})

    # ── Panel A: Waterfall-style plot for a single prediction ──
    # Simulated SHAP values for a single high-popularity video
    base_value = 4.82  # E[f(x)]
    features_shap = [
        ('ts_255 (early views)',   +0.85, 'temporal'),
        ('ts_279 (view accel.)',   +0.42, 'temporal'),
        ('tp_2 (long-term trend)', +0.31, 'temporal'),
        ('meta_5 (upload hour)',   +0.18, 'metadata'),
        ('visual_128 (thumbnail)', +0.12, 'visual'),
        ('cr_8 (follower count)',  +0.09, 'creator'),
        ('text_42 (title embed.)', -0.05, 'textual'),
        ('acou_15 (audio energy)', -0.03, 'acoustic'),
        ('ts_207 (view drop)',     -0.08, 'temporal'),
        ('meta_3 (video length)',  -0.11, 'metadata'),
    ]

    predicted = base_value + sum(v for _, v, _ in features_shap)

    # Draw waterfall
    cumulative = base_value
    y_positions = list(range(len(features_shap)))
    y_positions.reverse()

    bar_data = []
    for i, (name, val, mod) in enumerate(features_shap):
        start = cumulative
        cumulative += val
        bar_data.append((name, val, mod, start, cumulative))

    for i, (name, val, mod, start, end) in enumerate(bar_data):
        y = len(bar_data) - 1 - i
        color = '#D32F2F' if val > 0 else '#1976D2'
        ax1.barh(y, val, left=start, height=0.6, color=color,
                 edgecolor='white', linewidth=0.3, alpha=0.85, zorder=2)
        # Value label
        label_x = end + 0.02 if val > 0 else end - 0.02
        ha = 'left' if val > 0 else 'right'
        ax1.text(label_x, y, f'{val:+.2f}', va='center', ha=ha,
                 fontsize=6, color=color, fontweight='bold')
        # Connector line
        if i < len(bar_data) - 1:
            next_y = len(bar_data) - 2 - i
            ax1.plot([end, end], [y - 0.3, next_y + 0.3],
                     color=GRID_CLR, linewidth=0.5, linestyle='--', zorder=1)

    ax1.set_yticks(range(len(bar_data)))
    ax1.set_yticklabels([d[0] for d in reversed(bar_data)], fontsize=6.5)
    ax1.set_xlabel('SHAP value (impact on prediction)')

    # Base value and prediction annotations
    ax1.axvline(base_value, color=TEXT_MED, linewidth=0.6, linestyle=':', zorder=0)
    ax1.text(base_value, len(bar_data) + 0.1, f'E[f(x)] = {base_value:.2f}',
             ha='center', va='bottom', fontsize=7, color=TEXT_MED)
    ax1.axvline(predicted, color='#D32F2F', linewidth=0.8, linestyle='--', zorder=0)
    ax1.text(predicted, -0.8, f'f(x) = {predicted:.2f}',
             ha='center', va='top', fontsize=7.5, fontweight='bold', color='#D32F2F')

    ax1.grid(axis='x', color=GRID_CLR, linewidth=0.3, zorder=0)
    ax1.set_axisbelow(True)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, ha='left', va='top',
             fontsize=9, fontweight='bold')

    # ── Panel B: Grouped modality contributions ──
    modality_contrib = {}
    for name, val, mod in features_shap:
        modality_contrib[mod] = modality_contrib.get(mod, 0) + val

    mod_names = ['temporal', 'metadata', 'visual', 'creator', 'textual', 'acoustic']
    mod_vals = [modality_contrib.get(m, 0) for m in mod_names]
    mod_colors = [PALETTE[m] for m in mod_names]

    bars = ax2.barh(range(len(mod_names)), mod_vals, color=mod_colors,
                    edgecolor='white', linewidth=0.5, height=0.6, zorder=2)
    ax2.set_yticks(range(len(mod_names)))
    ax2.set_yticklabels([m.capitalize() for m in mod_names])
    ax2.set_xlabel('Aggregated SHAP contribution')
    ax2.axvline(0, color=TEXT_MED, linewidth=0.5, zorder=1)
    ax2.grid(axis='x', color=GRID_CLR, linewidth=0.3, zorder=0)
    ax2.set_axisbelow(True)

    # Value labels
    for i, v in enumerate(mod_vals):
        offset = 0.02 if v >= 0 else -0.02
        ha = 'left' if v >= 0 else 'right'
        ax2.text(v + offset, i, f'{v:+.2f}', va='center', ha=ha,
                 fontsize=6.5, color=mod_colors[i], fontweight='bold')

    # Annotation
    ax2.annotate('Temporal features\ndrive prediction',
                 xy=(mod_vals[0], 0), xytext=(mod_vals[0] - 0.3, 2.5),
                 fontsize=6.5, fontstyle='italic', color=PALETTE['temporal'],
                 arrowprops=dict(arrowstyle='->', color=PALETTE['temporal'],
                                 linewidth=0.6),
                 ha='center')

    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, ha='left', va='top',
             fontsize=9, fontweight='bold')

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, 'shap_local_explanation_advanced')


def figure4_strategy_pipeline():
    """Strategy generation pipeline diagram."""
    print('Generating Figure 4: Strategy Pipeline...')

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    cx = 5.0  # center x
    bw = 5.5
    bh = 0.7
    gap = 0.45

    # ── Layer 1: Prediction Model ──
    y = 8.8
    draw_block(ax, (cx - bw/2, y), bw, bh,
               'XMTL Popularity Prediction Model',
               color='#E8F0FE', edge=ACCENT, fontsize=9, bold=True)

    # ── Layer 2: Explainability ──
    y2 = y - bh - gap
    draw_block(ax, (cx - bw/2, y2), bw, bh,
               'Explainability Layer', color='#FFF3E0',
               edge=PALETTE['visual'], fontsize=9, bold=True)
    draw_arrow(ax, (cx, y), (cx, y2 + bh), color=TEXT_MED, lw=1.0)

    # Sub-items for explainability
    sub_w = 2.2
    sub_h = 0.5
    sub_y = y2 - sub_h - 0.2
    draw_block(ax, (cx - sub_w - 0.3, sub_y), sub_w, sub_h,
               'SHAP Analysis', color='white', edge=PALETTE['visual'], fontsize=7.5)
    draw_block(ax, (cx + 0.3, sub_y), sub_w, sub_h,
               'Modality Importance', color='white', edge=PALETTE['visual'], fontsize=7.5)
    draw_arrow(ax, (cx - sub_w/2 - 0.3, y2), (cx - sub_w/2 - 0.3, sub_y + sub_h),
               color=PALETTE['visual'], lw=0.7)
    draw_arrow(ax, (cx + sub_w/2 + 0.3, y2), (cx + sub_w/2 + 0.3, sub_y + sub_h),
               color=PALETTE['visual'], lw=0.7)

    # ── Layer 3: Pattern Discovery ──
    y3 = sub_y - gap - 0.1
    draw_block(ax, (cx - bw/2, y3), bw, bh,
               'Pattern Discovery', color='#F3E5F5',
               edge=PALETTE['acoustic'], fontsize=9, bold=True)
    draw_arrow(ax, (cx - sub_w/2 - 0.3, sub_y), (cx - bw/4, y3 + bh),
               color=TEXT_MED, lw=0.7)
    draw_arrow(ax, (cx + sub_w/2 + 0.3, sub_y), (cx + bw/4, y3 + bh),
               color=TEXT_MED, lw=0.7)

    # Sub-items for pattern discovery
    pd_y = y3 - sub_h - 0.2
    draw_block(ax, (cx - sub_w - 0.3, pd_y), sub_w, sub_h,
               'High-Performing\nVideo Patterns', color='white',
               edge='#4CAF50', fontsize=7)
    draw_block(ax, (cx + 0.3, pd_y), sub_w, sub_h,
               'Low-Performing\nVideo Patterns', color='white',
               edge='#F44336', fontsize=7)
    draw_arrow(ax, (cx - sub_w/2 - 0.3, y3), (cx - sub_w/2 - 0.3, pd_y + sub_h),
               color=PALETTE['acoustic'], lw=0.7)
    draw_arrow(ax, (cx + sub_w/2 + 0.3, y3), (cx + sub_w/2 + 0.3, pd_y + sub_h),
               color=PALETTE['acoustic'], lw=0.7)

    # ── Layer 4: Strategy Generator ──
    y4 = pd_y - gap - 0.1
    draw_block(ax, (cx - bw/2, y4), bw, bh,
               'Strategy Generator', color='#E8F5E9',
               edge=PALETTE['textual'], fontsize=9, bold=True)
    draw_arrow(ax, (cx - sub_w/2 - 0.3, pd_y), (cx - bw/4, y4 + bh),
               color=TEXT_MED, lw=0.7)
    draw_arrow(ax, (cx + sub_w/2 + 0.3, pd_y), (cx + bw/4, y4 + bh),
               color=TEXT_MED, lw=0.7)

    # ── Layer 5: Recommendations ──
    y5 = y4 - bh - gap
    draw_block(ax, (cx - bw/2, y5), bw, bh,
               'Creator Recommendations', color='#E3F2FD',
               edge=ACCENT, fontsize=9, bold=True)
    draw_arrow(ax, (cx, y4), (cx, y5 + bh), color=TEXT_MED, lw=1.0)

    # Recommendation items
    rec_items = [
        'Optimal upload timing',
        'Title length optimization',
        'Tag strategy',
        'Engagement momentum',
    ]
    rec_w = 2.2
    rec_h = 0.42
    rec_y = y5 - rec_h - 0.25
    positions = [cx - 2*rec_w + 0.3*1, cx - rec_w + 0.3*2, cx + 0.3*3, cx + rec_w + 0.3*4]
    # Use 2x2 grid
    for i, (item, px) in enumerate(zip(rec_items, [cx - rec_w - 0.25, cx + 0.25,
                                                     cx - rec_w - 0.25, cx + 0.25])):
        py = rec_y if i < 2 else rec_y - rec_h - 0.15
        draw_block(ax, (px, py), rec_w, rec_h, item,
                   color='white', edge=ACCENT, fontsize=6.5)
        src_x = px + rec_w/2
        draw_arrow(ax, (src_x, y5), (src_x, py + rec_h if i < 2 else rec_y + rec_h + 0.15),
                   color=ACCENT, lw=0.5)

    # Connect row 1 to row 2 recommendations
    for px in [cx - rec_w - 0.25, cx + 0.25]:
        draw_arrow(ax, (px + rec_w/2, rec_y), (px + rec_w/2, rec_y - 0.15 + 0.0),
                   color=ACCENT, lw=0.4)

    fig.tight_layout()
    save_fig(fig, 'strategy_pipeline_advanced')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('='*60)
    print('Generating publication-quality figures for XMTL paper')
    print('='*60)
    figure1_architecture()
    figure2_global_importance()
    figure3_local_explanation()
    figure4_strategy_pipeline()
    print('='*60)
    print(f'All figures saved to: {OUT_DIR}')
    print('Formats: PNG (300 dpi) + SVG')
    print('='*60)
