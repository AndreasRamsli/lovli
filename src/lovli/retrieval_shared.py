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


def normalize_law_ref(value: str) -> str:
    """Normalize law references (e.g., lov/1999-03-26-17) for matching."""
    normalized = (value or "").strip().lower().strip("/")
    return normalized


def _resolve_cross_reference_law_id(
    cross_reference: str,
    law_ref_to_id: dict[str, str] | None,
    candidate_law_ids: set[str],
) -> str | None:
    """Resolve a cross-reference token to a law_id when possible."""
    token = normalize_law_ref(cross_reference)
    if not token:
        return None
    if token in candidate_law_ids:
        return token
    if law_ref_to_id and token in law_ref_to_id:
        return law_ref_to_id[token]
    return None


def _compute_law_stats(
    candidates: list[dict[str, Any]],
    max_weight: float,
) -> list[dict[str, Any]]:
    """Compute per-law count/avg/max/strength and deterministic tie-break stats."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        law_id = (candidate.get("law_id") or "").strip() or "__unknown__"
        grouped.setdefault(law_id, []).append(candidate)

    total_count = len(candidates)
    avg_weight = 1.0 - max_weight
    law_stats: list[dict[str, Any]] = []
    for law_id, rows in grouped.items():
        law_scores = [float(row.get("score", 0.0)) for row in rows]
        avg_score = sum(law_scores) / len(law_scores) if law_scores else 0.0
        max_score = max(law_scores) if law_scores else 0.0
        variance = (
            sum((score - avg_score) ** 2 for score in law_scores) / len(law_scores)
            if law_scores
            else 0.0
        )
        concentration = (len(rows) / total_count) if total_count else 0.0
        strength = (avg_weight * avg_score) + (max_weight * max_score)
        law_stats.append(
            {
                "law_id": law_id,
                "count": len(rows),
                "avg_score": avg_score,
                "max_score": max_score,
                "strength": strength,
                "score_variance": variance,
                "concentration": concentration,
                # Lower variance is better during ties; invert into [0, 1].
                "consistency": 1.0 / (1.0 + variance),
            }
        )
    law_stats.sort(
        key=lambda item: (
            item["count"],
            item["strength"],
            item["concentration"],
            item["consistency"],
            item["max_score"],
            item["avg_score"],
        ),
        reverse=True,
    )
    return law_stats


def build_law_cross_reference_affinity(
    candidates: list[dict[str, Any]],
    law_ref_to_id: dict[str, str] | None = None,
    settings: Any | None = None,
    dominant_law_id: str | None = None,
) -> dict[str, float]:
    """
    Build simple law affinity scores from dominant-law cross references.

    Returns an affinity in [0, 1] per law_id where:
    - 1.0 = dominant law or directly referenced by dominant-law sources
    - 0.5 = law that references dominant law
    - 0.0 = no observed structural relationship
    """
    if not candidates:
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        law_id = (candidate.get("law_id") or "").strip()
        if not law_id:
            continue
        grouped.setdefault(law_id, []).append(candidate)
    if not grouped:
        return {}

    if not dominant_law_id:
        if settings is not None:
            max_weight = float(getattr(settings, "law_coherence_max_score_weight", 0.6))
            law_stats = _compute_law_stats(candidates, max_weight=max_weight)
            dominant_law_id = law_stats[0]["law_id"] if law_stats else None
        else:
            law_stats: list[tuple[str, int, float]] = []
            for law_id, rows in grouped.items():
                max_score = max(float(row.get("score", 0.0)) for row in rows)
                law_stats.append((law_id, len(rows), max_score))
            law_stats.sort(key=lambda item: (item[1], item[2]), reverse=True)
            dominant_law_id = law_stats[0][0] if law_stats else None
    if not dominant_law_id or dominant_law_id not in grouped:
        return {law_id: 0.0 for law_id in grouped.keys()}

    candidate_law_ids = set(grouped.keys())

    referenced_by_dominant: set[str] = set()
    for row in grouped.get(dominant_law_id, []):
        for raw_ref in row.get("cross_references") or []:
            resolved = _resolve_cross_reference_law_id(raw_ref, law_ref_to_id, candidate_law_ids)
            if resolved and resolved != dominant_law_id:
                referenced_by_dominant.add(resolved)

    references_dominant: set[str] = set()
    for law_id, rows in grouped.items():
        if law_id == dominant_law_id:
            continue
        for row in rows:
            for raw_ref in row.get("cross_references") or []:
                resolved = _resolve_cross_reference_law_id(raw_ref, law_ref_to_id, candidate_law_ids)
                if resolved == dominant_law_id:
                    references_dominant.add(law_id)

    affinity: dict[str, float] = {}
    for law_id in candidate_law_ids:
        if law_id == dominant_law_id:
            affinity[law_id] = 1.0
        elif law_id in referenced_by_dominant:
            affinity[law_id] = 1.0
        elif law_id in references_dominant:
            affinity[law_id] = 0.5
        else:
            affinity[law_id] = 0.0
    return affinity


def build_law_coherence_decision(
    candidates: list[dict[str, Any]],
    settings: Any,
    law_affinity_by_id: dict[str, float] | None = None,
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
    law_stats = _compute_law_stats(candidates, max_weight=max_weight)

    dominant = law_stats[0]
    dominant_law_id = dominant["law_id"]
    dominant_count = int(dominant["count"])
    dominant_avg_score = float(dominant["avg_score"])
    dominant_max_score = float(dominant["max_score"])
    dominant_strength = float(dominant["strength"])
    dominant_concentration = float(dominant["concentration"])
    result.update(
        {
            "dominant_law_id": dominant_law_id,
            "dominant_avg_score": round(dominant_avg_score, 6),
            "dominant_max_score": round(dominant_max_score, 6),
            "dominant_strength": round(dominant_strength, 6),
            "dominant_law_count": dominant_count,
            "dominant_concentration": round(dominant_concentration, 6),
            "dominant_score_variance": round(float(dominant.get("score_variance", 0.0)), 6),
            "dominant_consistency": round(float(dominant.get("consistency", 0.0)), 6),
        }
    )

    min_law_count = int(getattr(settings, "law_coherence_min_law_count", 2))
    min_abs_gap = float(getattr(settings, "law_coherence_score_gap", 0.15))
    min_rel_gap = float(getattr(settings, "law_coherence_relative_gap", 0.05))
    concentration_threshold = float(
        getattr(settings, "law_coherence_dominant_concentration_threshold", 0.60)
    )
    affinity_map = law_affinity_by_id or {}
    has_affinity_signal = bool(affinity_map)

    drop_indices: set[int] = set()
    decisions: list[dict[str, Any]] = []
    for item in law_stats[1:]:
        law_id = item["law_id"]
        law_count = int(item["count"])
        law_avg_score = float(item["avg_score"])
        law_max_score = float(item["max_score"])
        law_strength = float(item["strength"])
        law_variance = float(item.get("score_variance", 0.0))
        law_concentration = float(item.get("concentration", 0.0))
        law_consistency = float(item.get("consistency", 0.0))
        law_affinity = float(affinity_map.get(law_id, 0.0))
        if (
            law_count == 1
            and dominant_concentration >= concentration_threshold
            and has_affinity_signal
            and law_affinity <= 0.0
        ):
            drop_indices.update(idx for idx, _item in grouped.get(law_id, []))
            decisions.append(
                {
                    "law_id": law_id,
                    "law_count": law_count,
                    "decision": "dropped_dominant_concentration",
                    "dominant_concentration": round(dominant_concentration, 6),
                    "concentration_threshold": concentration_threshold,
                    "cross_reference_affinity": law_affinity,
                }
            )
            continue
        if law_affinity >= 1.0 and law_count == 1:
            decisions.append(
                {
                    "law_id": law_id,
                    "law_count": law_count,
                    "decision": "kept_cross_reference_affinity",
                    "cross_reference_affinity": law_affinity,
                }
            )
            continue
        if law_count >= min_law_count:
            decisions.append(
                {
                    "law_id": law_id,
                    "law_count": law_count,
                    "decision": "kept_min_count",
                    "cross_reference_affinity": law_affinity,
                }
            )
            continue
        abs_gap = dominant_strength - law_strength
        relative_gap = (abs_gap / dominant_strength) if dominant_strength > 0 else 0.0
        if abs_gap < min_abs_gap and relative_gap < min_rel_gap:
            dominant_tie_break = (
                (0.50 * dominant_concentration)
                + (0.30 * float(affinity_map.get(dominant_law_id, 1.0)))
                + (0.20 * float(dominant.get("consistency", 0.0)))
            )
            foreign_tie_break = (
                (0.50 * law_concentration)
                + (0.30 * law_affinity)
                + (0.20 * law_consistency)
            )
            # Deterministic tie-break: keep foreign law only when it is genuinely
            # competitive on concentration/affinity/consistency, not only raw score gap.
            if foreign_tie_break + 1e-9 < dominant_tie_break:
                drop_indices.update(idx for idx, _item in grouped.get(law_id, []))
                decisions.append(
                    {
                        "law_id": law_id,
                        "law_count": law_count,
                        "decision": "dropped_tie_break",
                        "abs_gap": round(abs_gap, 6),
                        "relative_gap": round(relative_gap, 6),
                        "law_avg_score": round(law_avg_score, 6),
                        "law_max_score": round(law_max_score, 6),
                        "law_score_variance": round(law_variance, 6),
                        "law_concentration": round(law_concentration, 6),
                        "cross_reference_affinity": law_affinity,
                        "dominant_tie_break": round(dominant_tie_break, 6),
                        "foreign_tie_break": round(foreign_tie_break, 6),
                    }
                )
                continue
            decisions.append(
                {
                    "law_id": law_id,
                    "law_count": law_count,
                    "decision": "kept_gap_too_small",
                    "abs_gap": round(abs_gap, 6),
                    "relative_gap": round(relative_gap, 6),
                    "law_avg_score": round(law_avg_score, 6),
                    "law_max_score": round(law_max_score, 6),
                    "law_score_variance": round(law_variance, 6),
                    "law_concentration": round(law_concentration, 6),
                    "cross_reference_affinity": law_affinity,
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
                "law_score_variance": round(law_variance, 6),
                "law_concentration": round(law_concentration, 6),
                "cross_reference_affinity": law_affinity,
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


def build_law_aware_rank_fusion(
    candidates: list[dict[str, Any]],
    settings: Any,
    law_affinity_by_id: dict[str, float] | None = None,
    routing_alignment_by_id: dict[str, float] | None = None,
    dominant_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute deterministic law-aware rank-fusion scores for candidate docs.

    Candidate format:
      - index: int
      - law_id: str
      - score: float
    """
    if not candidates:
        return {"ranked": [], "law_strengths": {}, "diagnostics": {}}

    affinity_map = law_affinity_by_id or {}
    routing_map = routing_alignment_by_id or {}
    max_weight = float(getattr(settings, "law_coherence_max_score_weight", 0.6))
    law_stats = _compute_law_stats(candidates, max_weight=max_weight)
    law_strengths = {item["law_id"]: float(item["strength"]) for item in law_stats}
    max_strength = max(law_strengths.values()) if law_strengths else 1.0
    if max_strength <= 0:
        max_strength = 1.0

    law_count_map = {item["law_id"]: int(item["count"]) for item in law_stats}
    total_candidates = len(candidates)

    dominant_law_id = (
        (dominant_context or {}).get("dominant_law_id")
        or (law_stats[0]["law_id"] if law_stats else "")
    )

    w_doc = float(getattr(settings, "law_rank_fusion_weight_doc_score", 0.55))
    w_route = float(getattr(settings, "law_rank_fusion_weight_routing", 0.20))
    w_aff = float(getattr(settings, "law_rank_fusion_weight_affinity", 0.10))
    w_dom = float(getattr(settings, "law_rank_fusion_weight_dominance", 0.10))
    w_conc = float(getattr(settings, "law_rank_fusion_weight_concentration", 0.05))
    weight_sum = w_doc + w_route + w_aff + w_dom + w_conc
    if weight_sum <= 0:
        w_doc, w_route, w_aff, w_dom, w_conc = 1.0, 0.0, 0.0, 0.0, 0.0
        weight_sum = 1.0

    ranked: list[dict[str, Any]] = []
    for row in candidates:
        law_id = (row.get("law_id") or "").strip() or "__unknown__"
        doc_score = float(row.get("score", 0.0))
        law_strength_norm = float(law_strengths.get(law_id, 0.0)) / max_strength
        routing_alignment = float(routing_map.get(law_id, 0.0))
        affinity = float(affinity_map.get(law_id, 0.0))
        concentration = (law_count_map.get(law_id, 0) / total_candidates) if total_candidates else 0.0
        dominance = 1.0 if law_id == dominant_law_id else law_strength_norm

        fused = (
            (w_doc * doc_score)
            + (w_route * routing_alignment)
            + (w_aff * affinity)
            + (w_dom * dominance)
            + (w_conc * concentration)
        ) / weight_sum

        ranked.append(
            {
                **row,
                "fused_score": float(fused),
                "fusion_components": {
                    "doc_score": round(doc_score, 6),
                    "routing_alignment": round(routing_alignment, 6),
                    "cross_reference_affinity": round(affinity, 6),
                    "dominance_context": round(dominance, 6),
                    "law_concentration": round(concentration, 6),
                    "law_strength_normalized": round(law_strength_norm, 6),
                },
            }
        )

    ranked.sort(
        key=lambda item: (
            item.get("fused_score", 0.0),
            item.get("score", 0.0),
            item.get("index", -1),
        ),
        reverse=True,
    )
    return {
        "ranked": ranked,
        "law_strengths": law_strengths,
        "diagnostics": {
            "dominant_law_id": dominant_law_id,
            "weights": {
                "doc_score": w_doc,
                "routing_alignment": w_route,
                "cross_reference_affinity": w_aff,
                "dominance_context": w_dom,
                "law_concentration": w_conc,
            },
        },
    }


def apply_uncertainty_law_cap(
    ranked_rows: list[dict[str, Any]],
    law_strengths: dict[str, float],
    settings: Any,
    is_uncertain: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply uncertainty-driven temporary law cap to ranked rows."""
    diagnostics = {"applied": False, "reason": "disabled_or_not_uncertain"}
    if not bool(getattr(settings, "law_uncertainty_law_cap_enabled", True)):
        return ranked_rows, diagnostics
    if not is_uncertain:
        return ranked_rows, diagnostics

    law_order = sorted(
        law_strengths.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    cap_laws = max(1, int(getattr(settings, "law_uncertainty_law_cap_max_laws", 2)))
    if len(law_order) <= cap_laws:
        diagnostics["reason"] = "already_within_cap"
        return ranked_rows, diagnostics

    top_strength = float(law_order[0][1]) if law_order else 0.0
    second_strength = float(law_order[1][1]) if len(law_order) > 1 else 0.0
    max_gap = float(getattr(settings, "law_uncertainty_law_cap_max_gap", 0.05))
    if (top_strength - second_strength) > max_gap:
        diagnostics["reason"] = "strength_gap_not_tied"
        return ranked_rows, diagnostics

    allowed_laws = {law_id for law_id, _strength in law_order[:cap_laws]}
    capped = [row for row in ranked_rows if (row.get("law_id") or "").strip() in allowed_laws]
    if not capped:
        diagnostics["reason"] = "cap_removed_everything"
        return ranked_rows, diagnostics

    diagnostics = {
        "applied": True,
        "reason": "uncertain_near_tie_cap_applied",
        "allowed_laws": sorted(allowed_laws),
        "cap_laws": cap_laws,
        "top_strength": top_strength,
        "second_strength": second_strength,
    }
    return capped, diagnostics
