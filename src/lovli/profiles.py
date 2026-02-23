"""Versioned trust profiles and chat-history utilities."""

from __future__ import annotations

from typing import Any


TRUST_PROFILES: dict[str, dict[str, Any]] = {
    "balanced_v1": {
        "trust_profile_name": "balanced_v1",
        "trust_profile_version": "2026-02-17",
        "retrieval_k_initial": 22,
        "reranker_confidence_threshold": 0.35,
        "reranker_min_doc_score": 0.32,
        "reranker_ambiguity_min_gap": 0.05,
        "reranker_ambiguity_top_score_ceiling": 0.70,
        "law_routing_fallback_unfiltered": True,
        "law_routing_fallback_max_laws": 16,
        "law_routing_fallback_min_lexical_support": 0,
        "law_routing_stage1_min_docs": 3,
        "law_routing_stage1_min_top_score": 0.38,
        "law_coherence_dominant_concentration_threshold": 0.60,
        "law_rank_fusion_enabled": True,
        "law_rank_fusion_weight_doc_score": 0.63,
        "law_rank_fusion_weight_routing": 0.12,
        "law_rank_fusion_weight_affinity": 0.10,
        "law_rank_fusion_weight_dominance": 0.10,
        "law_rank_fusion_weight_concentration": 0.05,
        "law_uncertainty_law_cap_enabled": False,
    },
    "balanced_v2": {
        # Fix 1: top_ceiling raised 0.70→0.80 — calibration shows [0.6,0.8] bin
        #   has 93.75% observed precision; the old ceiling was incorrectly treating
        #   high-precision retrievals as ambiguous, driving fp_gate=0.300.
        # Fix 2: min_gap raised 0.05→0.12 — adjacent articles within the same
        #   chapter score with gaps of 0.00-0.02; the wider gap forces decisiveness
        #   and reduces boundary_level_b wrong-article-same-law mismatches (60.8%).
        # Fix 3: min_doc_score raised 0.32→0.42 — the [0.4,0.6] calibration bin
        #   has only 25% observed precision; raising the floor cuts that FP bucket.
        # Fix 4: dualpass enabled — reduces 65% routing uncertainty by giving the
        #   router a second pass over summary text for queries missing the first pass.
        "trust_profile_name": "balanced_v2",
        "trust_profile_version": "2026-02-23",
        "retrieval_k_initial": 22,
        "reranker_confidence_threshold": 0.35,
        "reranker_min_doc_score": 0.42,
        "reranker_ambiguity_min_gap": 0.12,
        "reranker_ambiguity_top_score_ceiling": 0.80,
        "law_routing_fallback_unfiltered": True,
        "law_routing_fallback_max_laws": 16,
        "law_routing_fallback_min_lexical_support": 0,
        "law_routing_stage1_min_docs": 3,
        "law_routing_stage1_min_top_score": 0.38,
        "law_routing_summary_dualpass_enabled": True,
        # Raised from 0.15 → 0.20 to ensure explicitly-named laws rank first
        # even when their embedding sim is marginal (common for husleieloven queries).
        "law_routing_direct_mention_bonus": 0.20,
        "law_coherence_dominant_concentration_threshold": 0.60,
        "law_rank_fusion_enabled": True,
        "law_rank_fusion_weight_doc_score": 0.63,
        "law_rank_fusion_weight_routing": 0.12,
        "law_rank_fusion_weight_affinity": 0.10,
        "law_rank_fusion_weight_dominance": 0.10,
        "law_rank_fusion_weight_concentration": 0.05,
        "law_uncertainty_law_cap_enabled": False,
    },
    "strict_v1": {
        "trust_profile_name": "strict_v1",
        "trust_profile_version": "2026-02-17",
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


def extract_chat_history(
    messages: list[dict[str, Any]],
    window_size: int = 6,
    exclude_current: bool = True,
) -> list[dict[str, str]]:
    """
    Extract chat history for query rewriting.

    Args:
        messages: List of message dicts from session state
        window_size: Number of messages to include (default: 6)
        exclude_current: If True, exclude the last message (current question)

    Returns:
        List of message dicts with 'role' and 'content' keys, filtered to user/assistant only
    """
    if not messages:
        return []

    # Exclude current message if requested
    # If exclude_current=True, always exclude the last message (even if it's the only one)
    messages_to_process = messages[:-1] if exclude_current else messages

    # Filter and format messages
    chat_history = []
    for msg in messages_to_process[-window_size:]:
        role = msg.get("role", "").lower()
        if role in ("user", "assistant"):
            content = msg.get("content", "").strip()
            if content:  # Skip empty messages
                chat_history.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

    return chat_history
