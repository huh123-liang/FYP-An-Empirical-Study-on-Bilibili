# FYP-An-Empirical-Study-on-Bilibili
An explainable multimodal temporal learning project for video popularity prediction on Bilibili, integrating visual, acoustic, textual, metadata, creator, and early engagement signals, with benchmarking across multiple models and SHAP-based strategy generation for creators.
# XMTL: Explainable Multimodal Temporal Learning for Video Popularity Prediction

An empirical study on Bilibili video popularity prediction using multimodal features and explainable AI techniques.

## Overview

XMTL is a multimodal framework that integrates six modalities (visual, acoustic, textual, metadata, creator, temporal) through dedicated encoders and temporal-aware gated fusion for video popularity prediction. The project includes a controlled comparison across six model families, SHAP-based explainability analysis, ablation studies, and a strategy generation module for content creators.

## Key Results

| Model | MAE | RMSE | Spearman | R² |
|---|---|---|---|---|
| Ridge (content only) | 1.3775 | 1.7874 | 0.6811 | 0.4229 |
| XMTL (ours) | 1.3463 | 1.7874 | 0.6804 | 0.4229 |
| Ridge (full) | 1.0903 | 1.4567 | 0.8227 | 0.6167 |
| MLP | 0.9278 | 1.2981 | 0.8468 | 0.6956 |
| Random Forest | 0.5931 | 0.8338 | 0.9517 | 0.8744 |
| **XGBoost** | **0.5091** | **0.7209** | **0.9612** | **0.9061** |

**Key findings:**
- Temporal engagement features within the first 72 hours account for 56.1% of SHAP importance and alone yield R²=0.905
- XGBoost outperforms deep multimodal fusion on pre-extracted tabular features
- XMTL suffers from encoder bottleneck and fusion weight collapse, detailed in the paper

## Project Structure

```
├── configs/
│   └── default.yaml              # Model & training configuration
├── data/
│   ├── raw/                      # Original .npy and .csv data files
│   └── processed/                # Train/val/test splits
├── src/
│   ├── preprocessing.py          # Data verification & split generation
│   ├── dataset.py                # PyTorch Dataset (XMTLDataset)
│   └── dataloader.py             # DataLoader factory
├── models/
│   ├── encoders.py               # Modality-specific encoders + hierarchical GRU
│   ├── fusion.py                 # Gated / Attention / Concat fusion modules
│   ├── xmtl.py                   # Main XMTL model
│   └── baseline_models.py        # Ridge, MLP, Random Forest, XGBoost
├── training/
│   ├── train.py                  # XMTL training entry point
│   └── trainer.py                # Trainer with early stopping & LR scheduling
├── evaluation/
│   ├── metrics.py                # MAE, RMSE, Spearman, R²
│   └── evaluate.py               # Standalone evaluation script
├── experiments/
│   ├── run_baselines.py          # Train all baseline models
│   └── ablation.py               # Leave-one-out & single-modality ablation
├── explainability/
│   └── shap_analysis.py          # SHAP TreeExplainer + fusion weight analysis
├── recommendation/
│   └── strategy_generator.py     # Creator strategy generation from SHAP/ablation
├── scripts/
│   └── generate_figures.py       # Visualization & figure generation
├── outputs/
│   ├── figures/                  # Generated plots (PNG/SVG)
│   ├── results/                  # Metrics, predictions, ablation, SHAP results
│   └── logs/                     # Timestamped training logs
├── paper.tex                     # Research paper (LaTeX)
└── references.bib                # Bibliography
```

## Requirements

- Python 3.8+
- PyTorch >= 1.12
- scikit-learn
- XGBoost
- SHAP
- NumPy, Pandas, SciPy
- Matplotlib
- PyYAML

## Dataset

The dataset comprises 50,791 Bilibili videos with 8,662-dimensional features across six modalities:

| Modality | Dimensions | Description |
|---|---|---|
| Visual | 4,096 | Pre-extracted CNN features |
| Acoustic | 2,688 | Audio representations |
| Textual | 1,538 | Title/description/tag embeddings |
| Metadata | 22 | Duration, upload time, category |
| Creator | 9 | Follower count, upload frequency, reputation |
| Temporal | 309 | 72-hour and 6-day engagement signals |

Prediction target: log-transformed view count (range 0.0–12.31).

Data files should be placed under `data/raw/`:
- `content_features.npy` — shape `[50791, 8353]`
- `temporal_data_target.npy` — shape `[50791, 310]`

## Usage

### 1. Data Preprocessing

```bash
python -m src.preprocessing --config configs/default.yaml
```

Verifies data integrity and generates train/val/test splits (80/10/10, seed=42).

### 2. Train XMTL

```bash
python -m training.train --config configs/default.yaml
```

### 3. Train Baselines

```bash
python -m experiments.run_baselines --config configs/default.yaml
```

Trains Ridge (content-only & full), MLP, Random Forest, and XGBoost.

### 4. Ablation Studies

```bash
python -m experiments.ablation --config configs/default.yaml
```

Runs leave-one-out and single-modality ablation using XGBoost.

### 5. SHAP Analysis

```bash
python -m explainability.shap_analysis --config configs/default.yaml
```

### 6. Strategy Generation

```bash
python -m recommendation.strategy_generator --config configs/default.yaml
```

### 7. Evaluation

```bash
python -m evaluation.evaluate --config configs/default.yaml
```

## XMTL Architecture

- **Encoders**: 2-layer MLPs per modality (GELU activation, dropout 0.3), projecting to 128-dim
- **Temporal Encoder**: Hierarchical GRU — short-term (72h, hidden=128) with triple pooling + long-term (6-day, hidden=64) with residual injection
- **Fusion**: Temporal-aware gated fusion with temperature-scaled softmax (τ=0.5)
- **Prediction Head**: 128 → 64 → 1, Huber loss (δ=1.0)
- **Total Parameters**: 2.19M

## License

This project is for academic research purposes.
