"""Document reranking logic — pure functions extracted from chain.py.

These functions take explicit parameters rather than reading from ``self``,
making them independently testable and reusable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .scoring import normalize_sigmoid_scores

logger = logging.getLogger(__name__)


def build_reranker_document_text(
    doc: Any,
    *,
    context_enrichment_enabled: bool = True,
    context_max_prefix_chars: int = 200,
) -> str:
    """Build reranker text with optional metadata prefix for better disambiguation.

    Pure equivalent of ``LegalRAGChain._build_reranker_document_text``.

    Args:
        doc: A LangChain Document or dict-like object.
        context_enrichment_enabled: Whether to prepend law/article metadata.
        context_max_prefix_chars: Maximum length of the metadata prefix.

    Returns:
        Text string to pass to the cross-encoder.
    """
    base_content = doc.page_content if hasattr(doc, "page_content") else str(doc)
    if not context_enrichment_enabled:
        return base_content

    metadata = doc.metadata if hasattr(doc, "metadata") else {}
    if not isinstance(metadata, dict):
        return base_content

    law_short_name = (metadata.get("law_short_name") or "").strip()
    law_title = (metadata.get("law_title") or "").strip()
    provision_title = (metadata.get("title") or "").strip()
    chapter_title = (metadata.get("chapter_title") or "").strip()
    article_id = (metadata.get("article_id") or "").strip()

    prefix_parts: list[str] = []
    if law_short_name:
        prefix_parts.append(f"Lov kortnavn: {law_short_name}")
    elif law_title:
        prefix_parts.append(f"Lov: {law_title}")
    if provision_title and provision_title.lower() != law_title.lower():
        prefix_parts.append(f"Tittel: {provision_title}")
    if chapter_title:
        prefix_parts.append(f"Kapittel: {chapter_title}")
    if article_id:
        prefix_parts.append(f"ID: {article_id}")

    if not prefix_parts:
        return base_content

    prefix = " | ".join(prefix_parts)
    max_prefix = max(40, int(context_max_prefix_chars))
    if len(prefix) > max_prefix:
        prefix = prefix[: max_prefix - 1].rstrip() + "..."
    return f"{prefix}\n\n{base_content}"


def rerank_documents(
    query: str,
    docs: list[Any],
    reranker: Any,
    top_k: int,
    *,
    context_enrichment_enabled: bool = True,
    context_max_prefix_chars: int = 200,
) -> tuple[list[Any], list[float]]:
    """Rerank retrieved documents using a cross-encoder.

    Pure equivalent of ``LegalRAGChain._rerank``.

    Args:
        query: User query string.
        docs: Documents to rerank.
        reranker: CrossEncoder instance (or None → returns docs as-is).
        top_k: Maximum number of documents to return.
        context_enrichment_enabled: Passed to ``build_reranker_document_text``.
        context_max_prefix_chars: Passed to ``build_reranker_document_text``.

    Returns:
        ``(reranked_docs, scores)`` sorted by descending score.
    """
    if not reranker or not docs:
        return docs, []

    pairs = [
        [
            query,
            build_reranker_document_text(
                doc,
                context_enrichment_enabled=context_enrichment_enabled,
                context_max_prefix_chars=context_max_prefix_chars,
            ),
        ]
        for doc in docs
    ]

    try:
        raw_scores = reranker.predict(pairs)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]
        else:
            raw_scores = list(raw_scores)
    except Exception as exc:
        logger.warning("Reranking failed, returning original order: %s", exc)
        return docs, []

    if len(raw_scores) != len(docs):
        logger.warning(
            "Score count mismatch: %s scores for %s docs, returning original order",
            len(raw_scores),
            len(docs),
        )
        return docs, []

    scores = normalize_sigmoid_scores(raw_scores)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
    reranked_scores = [score for _, score in scored_docs[:top_k]]
    return reranked_docs, reranked_scores


def filter_reranked_docs(
    docs: list[Any],
    scores: list[float],
    *,
    min_doc_score: float = 0.0,
    min_sources: int = 1,
) -> tuple[list[Any], list[float]]:
    """Filter reranked docs by per-document score with a floor on minimum sources.

    Pure equivalent of ``LegalRAGChain._apply_reranker_doc_filter``.

    Args:
        docs: Reranked documents.
        scores: Aligned scores for ``docs``.
        min_doc_score: Minimum score to keep a document.
        min_sources: Always keep at least this many documents.

    Returns:
        ``(filtered_docs, filtered_scores)`` with at least ``min_sources`` entries.
    """
    if not docs or not scores:
        return docs, scores

    effective_min = min(min_sources, len(docs))
    kept = [(doc, score) for doc, score in zip(docs, scores) if score >= min_doc_score]
    if len(kept) < effective_min:
        kept = list(zip(docs, scores))[:effective_min]

    filtered_docs = [doc for doc, _ in kept]
    filtered_scores = [score for _, score in kept]
    dropped = len(docs) - len(filtered_docs)
    if dropped > 0:
        logger.debug(
            "Dropped %s low-score reranked docs (min_doc_score=%.2f)",
            dropped,
            min_doc_score,
        )
    return filtered_docs, filtered_scores


def should_gate_answer(
    top_score: Optional[float],
    scores: Optional[list[float]] = None,
    *,
    confidence_threshold: float = 0.0,
    ambiguity_gating_enabled: bool = False,
    ambiguity_top_score_ceiling: float = 1.0,
    ambiguity_min_gap: float = 0.0,
) -> bool:
    """Check if answer should be gated due to low confidence.

    Pure equivalent of ``LegalRAGChain.should_gate_answer``.

    Args:
        top_score: Highest reranker score, or None if no reranking.
        scores: Full sorted score list for ambiguity gating.
        confidence_threshold: Gate if ``top_score`` is below this.
        ambiguity_gating_enabled: Enable ambiguity gap check.
        ambiguity_top_score_ceiling: Only apply ambiguity check when top score ≤ this.
        ambiguity_min_gap: Minimum required gap between top and second score.

    Returns:
        True when the answer should be withheld.
    """
    if top_score is None:
        return False
    if top_score < confidence_threshold:
        return True
    if (
        ambiguity_gating_enabled
        and scores
        and len(scores) >= 2
        and scores[0] <= ambiguity_top_score_ceiling
    ):
        if (scores[0] - scores[1]) < ambiguity_min_gap:
            return True
    return False
