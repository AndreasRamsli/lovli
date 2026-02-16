"""Shared retrieval/evaluation utilities to keep runtime and scripts in sync."""

from __future__ import annotations

import math
from typing import Any


def sigmoid(x: float) -> float:
    """Numerically safe sigmoid used for reranker logit normalization."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0


def normalize_sigmoid_scores(raw_scores: Any) -> list[float]:
    """
    Normalize reranker logits to [0, 1].

    Accepts lists, tuples, numpy arrays, or scalar numeric values.
    """
    if raw_scores is None:
        return []
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, (int, float)):
        raw_scores = [raw_scores]
    return [sigmoid(float(score)) for score in list(raw_scores)]


def matches_expected_source(cited_source: dict[str, Any], expected_source: dict[str, Any]) -> bool:
    """Check precise law-aware match with prefix-compatible article semantics."""
    cited_law = (cited_source.get("law_id") or "").strip()
    cited_article = (cited_source.get("article_id") or "").strip()
    expected_law = (expected_source.get("law_id") or "").strip()
    expected_article = (expected_source.get("article_id") or "").strip()
    if not expected_law or not expected_article:
        return False
    return cited_law == expected_law and cited_article.startswith(expected_article)


def build_law_coherence_decision(
    candidates: list[dict[str, Any]], settings: Any
) -> dict[str, Any]:
    """
    Compute law coherence filtering decisions over scored candidates.

    Candidate format:
      - law_id: str
      - score: float

    Returns a diagnostics-friendly decision dict containing:
      - drop_indices: set[int]
      - reason: str
      - removed_count: int
      - dominant_law_id, dominant_avg_score, dominant_max_score, dominant_strength
      - decisions: list[dict]
    """
    result: dict[str, Any] = {
        "reason": "missing_docs_or_scores",
        "removed_count": 0,
        "drop_indices": set(),
        "decisions": [],
    }
    if not candidates or len(candidates) < 2:
        result["reason"] = "insufficient_docs"
        return result

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, candidate in enumerate(candidates):
        law_id = (candidate.get("law_id") or "").strip() or "__unknown__"
        grouped.setdefault(law_id, []).append((idx, candidate))
    if len(grouped) <= 1:
        result["reason"] = "single_law_only"
        return result

    max_weight = float(getattr(settings, "law_coherence_max_score_weight", 0.6))
    avg_weight = 1.0 - max_weight
    law_stats: list[tuple[str, int, float, float, float]] = []
    for law_id, indexed_items in grouped.items():
        law_scores = [float(item.get("score", 0.0)) for _idx, item in indexed_items]
        avg_score = sum(law_scores) / len(law_scores) if law_scores else 0.0
        max_score = max(law_scores) if law_scores else 0.0
        strength = (avg_weight * avg_score) + (max_weight * max_score)
        law_stats.append((law_id, len(indexed_items), avg_score, max_score, strength))
    law_stats.sort(key=lambda item: (item[1], item[4], item[3], item[2]), reverse=True)

    dominant_law_id, _dominant_count, dominant_avg_score, dominant_max_score, dominant_strength = law_stats[0]
    result.update(
        {
            "dominant_law_id": dominant_law_id,
            "dominant_avg_score": round(dominant_avg_score, 6),
            "dominant_max_score": round(dominant_max_score, 6),
            "dominant_strength": round(dominant_strength, 6),
        }
    )

    min_law_count = int(getattr(settings, "law_coherence_min_law_count", 2))
    min_abs_gap = float(getattr(settings, "law_coherence_score_gap", 0.15))
    min_rel_gap = float(getattr(settings, "law_coherence_relative_gap", 0.05))

    drop_indices: set[int] = set()
    decisions: list[dict[str, Any]] = []
    for law_id, law_count, law_avg_score, law_max_score, law_strength in law_stats[1:]:
        if law_count >= min_law_count:
            decisions.append({"law_id": law_id, "law_count": law_count, "decision": "kept_min_count"})
            continue
        abs_gap = dominant_strength - law_strength
        relative_gap = (abs_gap / dominant_strength) if dominant_strength > 0 else 0.0
        if abs_gap < min_abs_gap and relative_gap < min_rel_gap:
            decisions.append(
                {
                    "law_id": law_id,
                    "law_count": law_count,
                    "decision": "kept_gap_too_small",
                    "abs_gap": round(abs_gap, 6),
                    "relative_gap": round(relative_gap, 6),
                    "law_avg_score": round(law_avg_score, 6),
                    "law_max_score": round(law_max_score, 6),
                }
            )
            continue
        drop_indices.update(idx for idx, _item in grouped.get(law_id, []))
        decisions.append(
            {
                "law_id": law_id,
                "law_count": law_count,
                "decision": "dropped",
                "abs_gap": round(abs_gap, 6),
                "relative_gap": round(relative_gap, 6),
                "law_avg_score": round(law_avg_score, 6),
                "law_max_score": round(law_max_score, 6),
            }
        )

    if not drop_indices:
        result.update({"reason": "no_laws_met_drop_criteria", "decisions": decisions})
        return result

    min_keep = int(getattr(settings, "law_coherence_min_keep", 1))
    min_sources_floor = min(max(min_keep, 1), len(candidates))
    if (len(candidates) - len(drop_indices)) < min_sources_floor:
        sorted_drop_indices = sorted(
            drop_indices,
            key=lambda idx: float(candidates[idx].get("score", 0.0)),
            reverse=True,
        )
        required_keep = min_sources_floor - (len(candidates) - len(drop_indices))
        keep_back = set(sorted_drop_indices[:required_keep])
        drop_indices = set(idx for idx in drop_indices if idx not in keep_back)

    result.update(
        {
            "reason": "filtered" if drop_indices else "kept_after_floor_adjustment",
            "drop_indices": drop_indices,
            "removed_count": len(drop_indices),
            "min_sources_floor": min_sources_floor,
            "decisions": decisions,
        }
    )
    return result
