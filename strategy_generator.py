"""
recommendation/strategy_generator.py
-------------------------------------
Content creation strategy generator based on explainability analysis.

Reads SHAP importance scores and ablation results to produce actionable
recommendations for Bilibili content creators.

Usage:
    cd LTF_code/
    python -m recommendation.strategy_generator --config configs/default.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils.config import load_config
from utils.logger import get_logger

MODALITY_DESCRIPTIONS = {
    "visual": "Visual features (thumbnails, video frames, visual quality)",
    "acoustic": "Audio features (background music, voice quality, sound effects)",
    "textual": "Text features (title, description, tags, keywords)",
    "metadata": "Metadata (upload time, duration, category, resolution)",
    "creator": "Creator profile (follower count, upload history, engagement rate)",
    "temporal": "Temporal patterns (view trends, early engagement velocity)",
}


def load_analysis_results(result_dir: str) -> Dict:
    """Load all available analysis results from the outputs directory.

    Returns:
        Dict with keys: shap, ablation, fusion_weights (each may be None).
    """
    results = {"shap": None, "ablation": None, "fusion_weights": None}

    shap_path = os.path.join(result_dir, "explainability", "xgboost_shap.json")
    if os.path.isfile(shap_path):
        with open(shap_path) as f:
            results["shap"] = json.load(f)

    ablation_path = os.path.join(result_dir, "ablation", "xgboost_ablation.csv")
    if os.path.isfile(ablation_path):
        results["ablation"] = pd.read_csv(ablation_path)

    fusion_path = os.path.join(result_dir, "explainability", "fusion_weight_analysis.json")
    if os.path.isfile(fusion_path):
        with open(fusion_path) as f:
            results["fusion_weights"] = json.load(f)

    return results


def rank_modalities(results: Dict) -> List[Dict]:
    """Rank modalities by importance using available evidence.

    Combines SHAP importance percentages and ablation R² drops.

    Returns:
        Sorted list of dicts: [{modality, shap_pct, ablation_drop, combined_rank}]
    """
    modalities = list(MODALITY_DESCRIPTIONS.keys())
    scores = {m: {"modality": m, "shap_pct": 0.0, "ablation_drop": 0.0} for m in modalities}

    # SHAP importance
    if results["shap"] and "modality_importance_pct" in results["shap"]:
        for mod, pct in results["shap"]["modality_importance_pct"].items():
            if mod in scores:
                scores[mod]["shap_pct"] = pct

    # Ablation R² drop (leave-one-out)
    if results["ablation"] is not None:
        loo = results["ablation"][results["ablation"]["study"] == "leave_one_out"]
        full_row = results["ablation"][results["ablation"]["study"] == "full"]
        if not full_row.empty:
            full_r2 = full_row.iloc[0]["r2"]
            for _, row in loo.iterrows():
                mod = row["modality"]
                if mod in scores:
                    scores[mod]["ablation_drop"] = full_r2 - row["r2"]

    # Combined score: normalize both to [0,1] and average
    ranking = list(scores.values())
    max_shap = max(s["shap_pct"] for s in ranking) or 1.0
    max_drop = max(abs(s["ablation_drop"]) for s in ranking) or 1.0

    for s in ranking:
        norm_shap = s["shap_pct"] / max_shap
        norm_drop = abs(s["ablation_drop"]) / max_drop
        s["combined_score"] = (norm_shap + norm_drop) / 2

    ranking.sort(key=lambda x: x["combined_score"], reverse=True)
    for i, s in enumerate(ranking):
        s["rank"] = i + 1

    return ranking


def generate_strategies(ranking: List[Dict], results: Dict, logger) -> List[Dict]:
    """Generate actionable content creation strategies.

    Args:
        ranking: Modality ranking from rank_modalities().
        results: Analysis results dict.
        logger: Logger instance.

    Returns:
        List of strategy dicts.
    """
    strategies = []

    logger.info("\n" + "=" * 60)
    logger.info("CONTENT CREATION STRATEGIES")
    logger.info("=" * 60)

    # Strategy 1: Focus on top modalities
    top_3 = ranking[:3]
    logger.info(f"\n[Strategy 1] Focus on the top-{len(top_3)} most impactful modalities:")
    for item in top_3:
        mod = item["modality"]
        desc = MODALITY_DESCRIPTIONS[mod]
        logger.info(f"  #{item['rank']} {mod:12s} (SHAP: {item['shap_pct']:.1f}%, "
                     f"ablation drop: {item['ablation_drop']:+.4f} R²)")
        logger.info(f"      → {desc}")

    strategies.append({
        "id": 1,
        "title": "Prioritize high-impact modalities",
        "description": f"Focus content optimization on: {', '.join(m['modality'] for m in top_3)}",
        "modalities": [m["modality"] for m in top_3],
    })

    # Strategy 2: Temporal engagement patterns
    temporal_rank = next((r for r in ranking if r["modality"] == "temporal"), None)
    if temporal_rank:
        logger.info(f"\n[Strategy 2] Temporal engagement optimization:")
        logger.info(f"  Temporal features rank #{temporal_rank['rank']} "
                     f"(SHAP: {temporal_rank['shap_pct']:.1f}%)")
        logger.info("  → Early view velocity in the first 72 hours strongly predicts long-term popularity")
        logger.info("  → Optimize upload timing and initial promotion to maximize early engagement")

    strategies.append({
        "id": 2,
        "title": "Optimize early engagement velocity",
        "description": "Early view patterns (first 72 hours) are strong predictors. "
                       "Focus on upload timing and initial promotion.",
    })

    # Strategy 3: Content quality vs. metadata
    content_mods = [r for r in ranking if r["modality"] in ["visual", "acoustic", "textual"]]
    meta_mods = [r for r in ranking if r["modality"] in ["metadata", "creator"]]
    content_score = sum(m["combined_score"] for m in content_mods)
    meta_score = sum(m["combined_score"] for m in meta_mods)

    logger.info(f"\n[Strategy 3] Content quality vs. metadata/creator influence:")
    logger.info(f"  Content modalities (visual+acoustic+textual): combined score = {content_score:.3f}")
    logger.info(f"  Meta modalities (metadata+creator):           combined score = {meta_score:.3f}")

    if content_score > meta_score:
        logger.info("  → Content quality matters more than creator reputation or metadata optimization")
        strategies.append({
            "id": 3,
            "title": "Invest in content quality",
            "description": "Visual, audio, and textual quality outweigh metadata and creator effects.",
        })
    else:
        logger.info("  → Creator reputation and metadata optimization are highly influential")
        strategies.append({
            "id": 3,
            "title": "Leverage creator profile and metadata",
            "description": "Creator reputation and metadata (timing, tags) are key drivers.",
        })

    # Strategy 4: Top specific features
    if results["shap"] and "top_20_features" in results["shap"]:
        top_feats = results["shap"]["top_20_features"][:5]
        logger.info(f"\n[Strategy 4] Top 5 most predictive individual features:")
        for feat in top_feats:
            logger.info(f"  {feat['feature']:30s}: SHAP = {feat['mean_abs_shap']:.4f}")

        strategies.append({
            "id": 4,
            "title": "Optimize top predictive features",
            "description": "Focus on the specific features with highest SHAP importance.",
            "features": [f["feature"] for f in top_feats],
        })

    return strategies


# ============================================================================
# Main
# ============================================================================

def main(config_path: str):
    cfg = load_config(config_path)
    logger = get_logger("strategy", log_dir=cfg.output.log_dir)

    result_dir = cfg.output.result_dir
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(_PROJECT_DIR, result_dir)
    save_dir = os.path.join(result_dir, "recommendations")
    os.makedirs(save_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Strategy Generator")
    logger.info("=" * 60)

    # Load analysis results
    results = load_analysis_results(result_dir)
    available = [k for k, v in results.items() if v is not None]
    logger.info(f"Available analysis results: {available}")

    if not available:
        logger.error("No analysis results found. Run ablation and/or SHAP analysis first.")
        return

    # Rank modalities
    ranking = rank_modalities(results)
    logger.info("\nModality Ranking:")
    logger.info(f"  {'Rank':>4s} {'Modality':12s} {'SHAP%':>8s} {'Abl.Drop':>10s} {'Score':>8s}")
    logger.info("  " + "-" * 46)
    for r in ranking:
        logger.info(f"  {r['rank']:4d} {r['modality']:12s} {r['shap_pct']:7.1f}% "
                     f"{r['ablation_drop']:+9.4f} {r['combined_score']:8.3f}")

    # Generate strategies
    strategies = generate_strategies(ranking, results, logger)

    # Save
    output = {
        "modality_ranking": ranking,
        "strategies": strategies,
    }
    with open(os.path.join(save_dir, "strategies.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nStrategies saved to {save_dir}/strategies.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content Strategy Generator")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
