"""Shared dataclasses used across routing, reranking, and chain modules.

No imports from other lovli modules — keeps this safe as a foundation
that any module can import without risk of circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RoutingResult:
    """Result of law routing for a single query.

    Replaces ad-hoc mutation of ``LegalRAGChain._last_routing_diagnostics``.
    The ``diagnostics`` field preserves full backward compatibility for the
    sweep script and any caller that inspects routing internals.
    """

    law_ids: list[str]
    """Ordered list of routed law IDs (most confident first)."""

    is_uncertain: bool
    """True when routing confidence is below the uncertainty threshold."""

    fallback_stage: Optional[str]
    """Stage name if retrieval fell back to unfiltered, otherwise None."""

    fallback_triggered: bool
    """True when the unfiltered fallback path was used."""

    alignment_by_law: dict[str, float]
    """Per-law routing alignment score in [0, 1], used by rank fusion."""

    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Full diagnostics dict — kept for backward compat with sweep script."""


@dataclass
class RetrievalResult:
    """Result of a full retrieve() call, bundling docs, scores, and routing."""

    sources: list[dict[str, Any]]
    """Extracted, deduplicated source dicts ready for the LLM context."""

    top_score: Optional[float]
    """Highest reranker score across all sources, or None if no reranking."""

    scores: list[float]
    """Per-source reranker scores aligned with sources list."""

    routing: Optional[RoutingResult] = None
    """Routing result for this query, if routing was enabled."""
