"""
Regenerate Figure 1 (Strategy Pipeline) and Figure 2 (Architecture)
Publication-quality for NeurIPS / IEEE / KDD.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Global Style ──────────────────────────────────────────────────────────────

MOD_COLORS = {
    'temporal':  '#2171B5',
    'visual':    '#E6550D',
    'textual':   '#31A354',
    'acoustic':  '#756BB1',
    'metadata':  '#636363',
    'creator':   '#8B4513',
}

C_ARROW   = '#333333'
C_BORDER  = '#333333'
C_PRED    = '#1A1A1A'
C_FUSION  = '#B22222'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'cm',
    'text.usetex': False,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'figures')
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=300, facecolor='white')
    fig.savefig(os.path.join(OUT, f'{name}.svg'), facecolor='white')
    plt.close(fig)
    print(f'  -> {name}.png / .svg')

# ── Drawing primitives ────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, fc, fontsize=6.5, tc='white',
        bold=False, lw=0.6, ec=None, alpha=1.0, zorder=3, pad=0.015):
    if ec is None:
        ec = C_BORDER
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        alpha=alpha, zorder=zorder)
    ax.add_patch(b)
    fw = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fw, color=tc, zorder=zorder+1)

def arr(ax, x1, y1, x2, y2, color=C_ARROW, lw=0.7, style='->', ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=1, shrinkB=1), zorder=2)

def region(ax, x, y, w, h, label, color, fontsize=6.5, pos='top-center'):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                        facecolor=color + '08', edgecolor=color,
                        linewidth=0.9, linestyle=(0, (4, 3)), zorder=0.5)
    ax.add_patch(r)
    if pos == 'top-center':
        ax.text(x + w/2, y + h + 0.06, label, ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold', color=color)
    elif pos == 'top-left':
        ax.text(x + 0.1, y + h + 0.06, label, ha='left', va='bottom',
                fontsize=fontsize, fontweight='bold', color=color)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: STRATEGY PIPELINE (5-layer, detailed)
# ══════════════════════════════════════════════════════════════════════════════

def fig_strategy_pipeline():
    print('Figure: Strategy Pipeline...')

    fig, ax = plt.subplots(figsize=(7.16, 7.2))
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(2.0, 10.6)
    ax.set_aspect('equal')
    ax.axis('off')

    cx = 5.0  # center x
    main_w = 6.0
    main_h = 0.48
    sub_w = 2.6
    sub_h = 0.40

    # Spacing parameters
    gap_main_to_sub = 0.60   # main bar → its sub-items
    gap_sub_to_next = 0.93   # sub-items → next main bar
    gap_l1_l2 = 0.80        # L1 → L2

    # Layer colors: (fill, edge)
    L = {
        1: ('#E8F0FE', '#2171B5'),   # prediction
        2: ('#FFF3E0', '#E6550D'),   # explainability
        3: ('#F3E5F5', '#756BB1'),   # pattern discovery
        4: ('#E8F5E9', '#31A354'),   # strategy generation
        5: ('#E3F2FD', '#2171B5'),   # recommendations
    }

    def layer_label(y, num, text):
        ax.text(-0.05, y + main_h/2, f'L{num}', ha='right', va='center',
                fontsize=6, color='#999999', fontstyle='italic')

    # ── Layer 1: Prediction Model ──
    y1 = 9.8
    box(ax, cx - main_w/2, y1, main_w, main_h,
        'XMTL Popularity Prediction Model', L[1][0], fontsize=8,
        tc=L[1][1], bold=True, ec=L[1][1], lw=1.0)
    layer_label(y1, 1, 'Prediction')

    # ── Layer 2: Explainability ──
    y2 = y1 - main_h - gap_l1_l2
    box(ax, cx - main_w/2, y2, main_w, main_h,
        'Explainability Layer', L[2][0], fontsize=8,
        tc=L[2][1], bold=True, ec=L[2][1], lw=1.0)
    layer_label(y2, 2, 'Explain')
    arr(ax, cx, y1, cx, y2 + main_h, color=C_ARROW, lw=0.9)

    # Sub-items for L2
    y2s = y2 - gap_main_to_sub
    items_l2 = [
        ('SHAP Value\nComputation', cx - sub_w - 0.3),
        ('Modality Importance\nAggregation', cx + 0.3),
    ]
    for label, sx in items_l2:
        box(ax, sx, y2s, sub_w, sub_h, label, 'white', fontsize=6.5,
            tc='#333333', ec=L[2][1], lw=0.5)
    arr(ax, cx - sub_w/2 - 0.3, y2, cx - sub_w/2 - 0.3, y2s + sub_h,
        color=L[2][1], lw=0.6)
    arr(ax, cx + sub_w/2 + 0.3, y2, cx + sub_w/2 + 0.3, y2s + sub_h,
        color=L[2][1], lw=0.6)
    # Connecting line between sub-items
    arr(ax, cx - 0.3 + 0.02, y2s + sub_h/2, cx + 0.3 - 0.02, y2s + sub_h/2,
        color='#CCCCCC', lw=0.4, style='-')

    # ── Layer 3: Pattern Discovery ──
    y3 = y2s - gap_sub_to_next
    box(ax, cx - main_w/2, y3, main_w, main_h,
        'Pattern Discovery', L[3][0], fontsize=8,
        tc=L[3][1], bold=True, ec=L[3][1], lw=1.0)
    layer_label(y3, 3, 'Pattern')
    arr(ax, cx - sub_w/2 - 0.3, y2s, cx - main_w/4, y3 + main_h,
        color=C_ARROW, lw=0.6)
    arr(ax, cx + sub_w/2 + 0.3, y2s, cx + main_w/4, y3 + main_h,
        color=C_ARROW, lw=0.6)

    # Sub-items for L3
    y3s = y3 - gap_main_to_sub
    box(ax, cx - sub_w - 0.3, y3s, sub_w, sub_h,
        'High-Performing\nVideo Clustering', 'white', fontsize=6.5,
        tc='#333333', ec='#4CAF50', lw=0.5)
    box(ax, cx + 0.3, y3s, sub_w, sub_h,
        'Low-Performing\nVideo Clustering', 'white', fontsize=6.5,
        tc='#333333', ec='#E53935', lw=0.5)
    arr(ax, cx - sub_w/2 - 0.3, y3, cx - sub_w/2 - 0.3, y3s + sub_h,
        color=L[3][1], lw=0.6)
    arr(ax, cx + sub_w/2 + 0.3, y3, cx + sub_w/2 + 0.3, y3s + sub_h,
        color=L[3][1], lw=0.6)

    # ── Layer 4: Strategy Generation ──
    y4 = y3s - gap_sub_to_next
    box(ax, cx - main_w/2, y4, main_w, main_h,
        'Strategy Generator', L[4][0], fontsize=8,
        tc=L[4][1], bold=True, ec=L[4][1], lw=1.0)
    layer_label(y4, 4, 'Strategy')
    arr(ax, cx - sub_w/2 - 0.3, y3s, cx - main_w/4, y4 + main_h,
        color=C_ARROW, lw=0.6)
    arr(ax, cx + sub_w/2 + 0.3, y3s, cx + main_w/4, y4 + main_h,
        color=C_ARROW, lw=0.6)

    # Sub-items for L4
    y4s = y4 - gap_main_to_sub
    box(ax, cx - sub_w - 0.3, y4s, sub_w, sub_h,
        'Rule Extraction', 'white', fontsize=6.5,
        tc='#333333', ec=L[4][1], lw=0.5)
    box(ax, cx + 0.3, y4s, sub_w, sub_h,
        'Counterfactual\nRecommendations', 'white', fontsize=6.5,
        tc='#333333', ec=L[4][1], lw=0.5)
    arr(ax, cx - sub_w/2 - 0.3, y4, cx - sub_w/2 - 0.3, y4s + sub_h,
        color=L[4][1], lw=0.6)
    arr(ax, cx + sub_w/2 + 0.3, y4, cx + sub_w/2 + 0.3, y4s + sub_h,
        color=L[4][1], lw=0.6)

    # ── Layer 5: Creator Recommendations ──
    y5 = y4s - gap_sub_to_next
    box(ax, cx - main_w/2, y5, main_w, main_h,
        'Creator Recommendations', L[5][0], fontsize=8,
        tc=L[5][1], bold=True, ec=L[5][1], lw=1.0)
    layer_label(y5, 5, 'Output')
    arr(ax, cx - sub_w/2 - 0.3, y4s, cx - main_w/4, y5 + main_h,
        color=C_ARROW, lw=0.6)
    arr(ax, cx + sub_w/2 + 0.3, y4s, cx + main_w/4, y5 + main_h,
        color=C_ARROW, lw=0.6)

    # Two recommendation groups
    y5s = y5 - 1.05
    grp_w = 2.6
    grp_h = 0.55

    # Content Optimization group
    gx_left = cx - grp_w - 0.3
    region(ax, gx_left - 0.08, y5s - 0.08, grp_w + 0.16, grp_h + 0.16,
           'Content Optimization', L[2][1], fontsize=5.5, pos='top-center')
    # Items inside
    item_h = 0.22
    box(ax, gx_left + 0.05, y5s + grp_h - item_h - 0.03, grp_w - 0.1, item_h,
        'Title length optimization', 'white', fontsize=5.5,
        tc='#333333', ec=L[2][1], lw=0.4, pad=0.008)
    box(ax, gx_left + 0.05, y5s + 0.03, grp_w - 0.1, item_h,
        'Tag strategy', 'white', fontsize=5.5,
        tc='#333333', ec=L[2][1], lw=0.4, pad=0.008)

    # Publishing Strategy group
    gx_right = cx + 0.3
    region(ax, gx_right - 0.08, y5s - 0.08, grp_w + 0.16, grp_h + 0.16,
           'Publishing Strategy', MOD_COLORS['temporal'], fontsize=5.5, pos='top-center')
    box(ax, gx_right + 0.05, y5s + grp_h - item_h - 0.03, grp_w - 0.1, item_h,
        'Optimal upload timing', 'white', fontsize=5.5,
        tc='#333333', ec=MOD_COLORS['temporal'], lw=0.4, pad=0.008)
    box(ax, gx_right + 0.05, y5s + 0.03, grp_w - 0.1, item_h,
        'Engagement momentum', 'white', fontsize=5.5,
        tc='#333333', ec=MOD_COLORS['temporal'], lw=0.4, pad=0.008)

    # Arrows from L5 to groups
    arr(ax, cx - main_w/4, y5, gx_left + grp_w/2, y5s + grp_h + 0.08 + 0.16,
        color=L[2][1], lw=0.5)
    arr(ax, cx + main_w/4, y5, gx_right + grp_w/2, y5s + grp_h + 0.08 + 0.16,
        color=MOD_COLORS['temporal'], lw=0.5)

    save(fig, 'strategy_pipeline_advanced')


def fig_architecture():
    print('Figure: Architecture...')

    fig, ax = plt.subplots(figsize=(7.16, 5.0))
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.6, 5.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Layout grid ──
    # Columns: Input | Encoder | (dim) | Fusion | Prediction | Output
    col_inp = 0.0
    col_enc = 1.85
    col_dim = 3.45
    col_fuse = 5.4
    col_pred = 7.6
    col_out  = 9.3

    bw = 1.4   # box width for input/encoder
    bh = 0.30  # box height
    ew = 1.2   # encoder width

    # ── Content modalities ──
    content_mods = [
        ('Visual',   'visual',   '4096', '256→128'),
        ('Acoustic', 'acoustic', '2688', '256→128'),
        ('Textual',  'textual',  '1538', '128→128'),
        ('Metadata', 'metadata', '22',   '64→128'),
        ('Creator',  'creator',  '9',    '64→128'),
    ]
    y_rows = [4.55, 4.05, 3.55, 3.05, 2.55]

    # Region: Content Representation Learning
    region(ax, col_inp - 0.12, 2.38, col_dim + 0.15, 2.72,
           'Content Representation Learning', '#555555', fontsize=6.5)

    for i, (name, key, dims, arch) in enumerate(content_mods):
        y = y_rows[i]
        c = MOD_COLORS[key]

        # Input box
        box(ax, col_inp, y, bw, bh, f'{name} ({dims}d)', c,
            fontsize=5.8, lw=0.5)
        # Arrow input → encoder (black)
        arr(ax, col_inp + bw + 0.02, y + bh/2,
            col_enc - 0.02, y + bh/2, color=C_ARROW, lw=0.5)
        # Encoder box
        box(ax, col_enc, y, ew, bh, f'MLP [{arch}]', c,
            fontsize=5.5, lw=0.5)
        # Arrow encoder → dim (black)
        arr(ax, col_enc + ew + 0.02, y + bh/2,
            col_dim + 0.15, y + bh/2, color=C_ARROW, lw=0.4)
        # Dimension label
        ax.text(col_dim + 0.22, y + bh/2, '128d', fontsize=5, va='center',
                color=c, fontstyle='italic')

    # Activation annotation
    ax.text(col_enc + ew/2, 2.42, 'GELU + Dropout(0.3)',
            ha='center', va='top', fontsize=5, color='#888888', fontstyle='italic')

    # ── Temporal Encoder ──
    ct = MOD_COLORS['temporal']
    ty = 1.3  # temporal row y

    # Region: Temporal Dynamics Modeling (emphasized)
    region(ax, col_inp - 0.12, 0.85, col_dim + 0.15, 1.15,
           'Temporal Dynamics Modeling', ct, fontsize=6.5)
    # Extra emphasis: thicker region border
    emphasis = FancyBboxPatch(
        (col_inp - 0.15, 0.82), col_dim + 0.21, 1.21,
        boxstyle="round,pad=0.04", facecolor=ct + '06', edgecolor=ct,
        linewidth=1.4, linestyle='-', zorder=0.3)
    ax.add_patch(emphasis)

    # Temporal input
    box(ax, col_inp, ty, bw, bh + 0.06,
        'Temporal (72h×4 + 6m×3)', ct, fontsize=5.5, lw=0.6)

    # Short-term GRU
    gru_x1 = col_enc - 0.05
    gru_w = 0.85
    arr(ax, col_inp + bw + 0.02, ty + (bh+0.06)/2,
        gru_x1 - 0.02, ty + 0.18, color=ct, lw=0.6)
    box(ax, gru_x1, ty + 0.01, gru_w, bh + 0.04,
        'GRU\n72h×4', ct, fontsize=5.5, lw=0.6)

    # Residual label
    res_mid = gru_x1 + gru_w + 0.12
    ax.text(res_mid, ty + 0.42, '+res', fontsize=4.5, ha='center',
            color=ct, fontstyle='italic')

    # Long-term GRU
    gru_x2 = res_mid + 0.12
    arr(ax, gru_x1 + gru_w + 0.01, ty + 0.18,
        gru_x2 - 0.01, ty + 0.18, color=ct, lw=0.5)
    box(ax, gru_x2, ty + 0.01, gru_w, bh + 0.04,
        'GRU\n6m×3', ct, fontsize=5.5, lw=0.6)

    # Arrow to dim
    arr(ax, gru_x2 + gru_w + 0.01, ty + 0.18,
        col_dim + 0.15, ty + 0.18, color=ct, lw=0.5)
    ax.text(col_dim + 0.22, ty + 0.18, '128d', fontsize=5, va='center',
            color=ct, fontstyle='italic', fontweight='bold')

    # ── Fusion module ──
    fuse_w = 1.3
    fuse_h = 3.0
    fuse_y = 1.0
    fuse_x = col_fuse

    # Region: Multimodal Fusion
    region(ax, fuse_x - 0.12, fuse_y - 0.15, fuse_w + 0.24, fuse_h + 0.35,
           'Multimodal Fusion', C_FUSION, fontsize=6.5)

    # Fusion block
    fb = FancyBboxPatch((fuse_x, fuse_y), fuse_w, fuse_h,
                         boxstyle="round,pad=0.03", facecolor=C_FUSION,
                         edgecolor=C_BORDER, linewidth=0.9, zorder=3, alpha=0.92)
    ax.add_patch(fb)
    ax.text(fuse_x + fuse_w/2, fuse_y + fuse_h/2 + 0.35,
            'Temporal-Aware\nGated Fusion', ha='center', va='center',
            fontsize=7, color='white', fontweight='bold', zorder=4)

    # Gating equations inside fusion block
    ax.text(fuse_x + fuse_w/2, fuse_y + fuse_h/2 - 0.15,
            r'$g_i = \sigma\!\left(\mathbf{W}_g[\mathbf{h}_i;\mathbf{t}]\right)$',
            ha='center', va='center', fontsize=6.5, color='#FFCCCC', zorder=4)
    ax.text(fuse_x + fuse_w/2, fuse_y + fuse_h/2 - 0.55,
            r'$w_i = \mathrm{softmax}(g_i)$',
            ha='center', va='center', fontsize=6.5, color='#FFCCCC', zorder=4)
    ax.text(fuse_x + fuse_w/2, fuse_y + fuse_h/2 - 0.95,
            r'$\mathbf{z} = \sum_i w_i \cdot \mathbf{h}_i$',
            ha='center', va='center', fontsize=6.5, color='#FFCCCC', zorder=4)

    # Arrows: content encoders → fusion (black, main data flow)
    for y in y_rows:
        arr(ax, col_dim + 0.55, y + bh/2,
            fuse_x, fuse_y + fuse_h/2, color=C_ARROW, lw=0.35)

    # Arrow: temporal → fusion (blue, emphasized)
    arr(ax, col_dim + 0.55, ty + 0.18,
        fuse_x, fuse_y + fuse_h/2, color=ct, lw=0.8)

    # ── Prediction Head ──
    pred_w = 1.2
    pred_h = 0.9
    pred_y = fuse_y + fuse_h/2 - pred_h/2
    pred_x = col_pred

    # Region: Prediction Head
    region(ax, pred_x - 0.08, pred_y - 0.12, pred_w + 0.16, pred_h + 0.24,
           'Prediction Head', C_PRED, fontsize=6.5)

    box(ax, pred_x, pred_y, pred_w, pred_h,
        'MLP\n128→64→1', C_PRED, fontsize=6.5, tc='white', bold=True, lw=0.8)

    # Arrow fusion → prediction (black)
    arr(ax, fuse_x + fuse_w, fuse_y + fuse_h/2,
        pred_x, pred_y + pred_h/2, color=C_ARROW, lw=0.9)

    # Output
    arr(ax, pred_x + pred_w, pred_y + pred_h/2,
        col_out, pred_y + pred_h/2, color=C_ARROW, lw=1.0)
    ax.text(col_out + 0.15, pred_y + pred_h/2,
            r'$\hat{y}$' + '\n(log views)',
            fontsize=7, va='center', color=C_PRED, fontweight='bold')

    # ── Explainability Branch (orange dashed) ──
    shap_color = '#E6550D'
    shap_y = pred_y - 1.3
    shap_w = 1.0
    shap_h = 0.45

    # SHAP Analysis box
    box(ax, pred_x + 0.1, shap_y, shap_w, shap_h,
        'SHAP\nAnalysis', shap_color, fontsize=6, tc='white', lw=0.6)
    # Dashed arrow from prediction to SHAP
    arr(ax, pred_x + pred_w/2, pred_y,
        pred_x + 0.1 + shap_w/2, shap_y + shap_h,
        color=shap_color, lw=0.7, ls='--')

    # Strategy Generator
    strat_x = col_out - 0.3
    strat_w = 1.0
    box(ax, strat_x, shap_y, strat_w, shap_h,
        'Strategy\nGenerator', '#31A354', fontsize=6, tc='white', lw=0.6)
    # Dashed arrow SHAP → Strategy
    arr(ax, pred_x + 0.1 + shap_w, shap_y + shap_h/2,
        strat_x, shap_y + shap_h/2,
        color=shap_color, lw=0.6, ls='--')

    # Region label for explainability branch
    region(ax, pred_x - 0.0, shap_y - 0.12, col_out - pred_x + 0.9, shap_h + 0.24,
           'Explainability Branch', shap_color, fontsize=5.5, pos='top-left')

    # ── Legend ──
    legend_y = 0.0
    legend_x = 0.0
    items = [
        ('Main data flow', C_ARROW, '-'),
        ('Temporal pathway', ct, '-'),
        ('Explainability', shap_color, '--'),
    ]
    for i, (label, color, ls) in enumerate(items):
        lx = legend_x + i * 2.8
        ax.plot([lx, lx + 0.4], [legend_y, legend_y],
                color=color, linewidth=1.0, linestyle=ls, zorder=5)
        ax.text(lx + 0.5, legend_y, label, fontsize=5.5, va='center',
                color='#555555')

    save(fig, 'architecture_advanced')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('='*55)
    print('Generating improved figures (v3)')
    print('='*55)
    fig_strategy_pipeline()
    fig_architecture()
    print('='*55)
    print(f'Saved to: {OUT}')
    print('='*55)
