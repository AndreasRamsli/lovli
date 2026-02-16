"""Versioned trust profiles for retrieval/routing behavior."""

from __future__ import annotations

from typing import Any


TRUST_PROFILES: dict[str, dict[str, Any]] = {
    "balanced_v1": {
        "trust_profile_name": "balanced_v1",
        "trust_profile_version": "2026-02-16",
        "retrieval_k_initial": 20,
        "reranker_confidence_threshold": 0.35,
        "reranker_min_doc_score": 0.35,
        "reranker_ambiguity_min_gap": 0.05,
        "reranker_ambiguity_top_score_ceiling": 0.70,
        "law_routing_fallback_unfiltered": True,
        "law_routing_stage1_min_docs": 2,
        "law_routing_stage1_min_top_score": 0.32,
        "law_coherence_dominant_concentration_threshold": 0.60,
        "law_rank_fusion_enabled": True,
        "law_rank_fusion_weight_doc_score": 0.55,
        "law_rank_fusion_weight_routing": 0.20,
        "law_rank_fusion_weight_affinity": 0.10,
        "law_rank_fusion_weight_dominance": 0.10,
        "law_rank_fusion_weight_concentration": 0.05,
    },
    "strict_v1": {
        "trust_profile_name": "strict_v1",
        "trust_profile_version": "2026-02-16",
        "retrieval_k_initial": 15,
        "reranker_confidence_threshold": 0.45,
        "reranker_min_doc_score": 0.55,
        "reranker_ambiguity_min_gap": 0.10,
        "reranker_ambiguity_top_score_ceiling": 0.70,
        "law_routing_fallback_unfiltered": False,
        "law_routing_stage1_min_docs": 3,
        "law_routing_stage1_min_top_score": 0.38,
        "law_coherence_dominant_concentration_threshold": 0.70,
        "law_rank_fusion_enabled": True,
        "law_rank_fusion_weight_doc_score": 0.50,
        "law_rank_fusion_weight_routing": 0.22,
        "law_rank_fusion_weight_affinity": 0.13,
        "law_rank_fusion_weight_dominance": 0.10,
        "law_rank_fusion_weight_concentration": 0.05,
    },
}


def apply_trust_profile(settings: Any, profile_name: str) -> str:
    """
    Apply a named trust profile to a settings object.

    Returns the resolved profile name (falls back to balanced_v1).
    """
    resolved = profile_name if profile_name in TRUST_PROFILES else "balanced_v1"
    profile = TRUST_PROFILES[resolved]
    for key, value in profile.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return resolved
