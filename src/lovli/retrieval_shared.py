"""Backward-compatibility shim — import from lovli.scoring instead."""

# ruff: noqa: F401
from .scoring import (
    sigmoid,
    normalize_sigmoid_scores,
    matches_expected_source,
    normalize_law_ref,
    _resolve_cross_reference_law_id,
    _compute_law_stats,
    build_law_cross_reference_affinity,
    build_law_coherence_decision,
    build_law_aware_rank_fusion,
    apply_uncertainty_law_cap,
    _infer_doc_type,
)
