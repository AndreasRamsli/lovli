"""Law routing logic — pure functions extracted from chain.py.

These functions take explicit parameters rather than reading from ``self``,
making them independently testable and reusable by the sweep script.
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np  # pragma: no cover

try:
    import numpy as _np_module  # type: ignore[import-untyped]

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _np_module = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

from .scoring import normalize_sigmoid_scores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_ROUTING_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_ROUTING_STOPWORDS = {
    "og",
    "som",
    "kan",
    "skal",
    "for",
    "fra",
    "med",
    "ved",
    "til",
    "av",
    "den",
    "det",
    "har",
    "hva",
    "nar",
    "eller",
    "ikke",
    "etter",
}


def tokenize_for_routing(text: str) -> set[str]:
    """Tokenize text for lightweight lexical routing."""
    tokens = set(_ROUTING_TOKEN_RE.findall((text or "").lower()))
    return {token for token in tokens if token not in _ROUTING_STOPWORDS}


def normalize_keyword_term(value: str) -> str:
    """Normalize chapter keywords to compact lexical routing terms."""
    normalized = re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def normalize_law_mention(text: str) -> str:
    """Normalize text for robust law short-name mention checks."""
    normalized = (text or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ---------------------------------------------------------------------------
# Routing-entry construction
# ---------------------------------------------------------------------------


def build_routing_entries(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Precompute routing text/tokens for fast hybrid law routing.

    Pure equivalent of ``LegalRAGChain._build_routing_entries``.
    """
    entries: list[dict[str, Any]] = []
    for item in catalog:
        law_id = (item.get("law_id") or "").strip()
        if not law_id:
            continue
        title = (item.get("law_title") or "").strip()
        short_name = (item.get("law_short_name") or "").strip()
        summary = (item.get("summary") or "").strip()
        law_ref = (item.get("law_ref") or "").strip()
        legal_area = (item.get("legal_area") or "").strip()
        chapter_titles = item.get("chapter_titles") or []
        if not isinstance(chapter_titles, list):
            chapter_titles = [str(chapter_titles)]
        chapter_keywords = item.get("chapter_keywords") or []
        if not isinstance(chapter_keywords, list):
            chapter_keywords = [str(chapter_keywords)]
        chapter_titles_text = " ".join(
            chapter_title.strip()
            for chapter_title in chapter_titles[:5]
            if isinstance(chapter_title, str) and chapter_title.strip()
        )
        chapter_keywords_text = " ".join(
            normalize_keyword_term(keyword)
            for keyword in chapter_keywords[:20]
            if isinstance(keyword, str) and normalize_keyword_term(keyword)
        )
        routing_title_text = " ".join(part for part in [title, short_name, law_ref] if part)
        routing_summary_text = " ".join(
            part
            for part in [summary, legal_area, chapter_titles_text, chapter_keywords_text]
            if part
        )
        routing_text = " ".join(
            part
            for part in [
                title,
                short_name,
                summary,
                law_ref,
                legal_area,
                chapter_titles_text,
                chapter_keywords_text,
            ]
            if part
        )
        entries.append(
            {
                "law_id": law_id,
                "law_title": title,
                "law_short_name": short_name,
                "routing_title_text": routing_title_text,
                "routing_summary_text": routing_summary_text,
                "routing_text": routing_text,
                "routing_tokens": tokenize_for_routing(routing_text),
                "chapter_keywords": chapter_keywords,
                "short_name_normalized": normalize_law_mention(short_name),
                "law_title_normalized": normalize_law_mention(title),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Lexical scoring
# ---------------------------------------------------------------------------


def score_law_candidates_lexical(
    query: str,
    catalog_entries: list[dict[str, Any]],
    min_token_overlap: int = 1,
    prefilter_k: int = 20,
) -> list[dict[str, Any]]:
    """Score catalog entries lexically.

    Pure equivalent of ``LegalRAGChain._score_law_candidates_lexical``.

    Args:
        query: The routing query text.
        catalog_entries: Precomputed routing entries from ``build_routing_entries``.
        min_token_overlap: Minimum token overlap to include a candidate.
        prefilter_k: Maximum number of candidates to return.

    Returns:
        Sorted list of candidate dicts with ``law_id``, ``lexical_score``, etc.
    """
    query_tokens = tokenize_for_routing(query)
    if not query_tokens:
        return []
    query_norm = normalize_law_mention(query)

    scored: list[dict[str, Any]] = []
    for entry in catalog_entries:
        overlap = len(query_tokens & entry["routing_tokens"])
        short_name = entry.get("short_name_normalized") or ""
        law_title = entry.get("law_title_normalized") or ""

        direct_mention = False
        if short_name and short_name in query_norm:
            direct_mention = True
            overlap += 5
        elif law_title and law_title in query_norm:
            direct_mention = True
            overlap += 5

        if overlap < min_token_overlap:
            continue

        scored.append(
            {
                "law_id": entry["law_id"],
                "law_title": entry.get("law_title", ""),
                "law_short_name": entry.get("law_short_name", ""),
                "routing_text": entry.get("routing_text", ""),
                "routing_title_text": entry.get("routing_title_text", ""),
                "routing_summary_text": entry.get("routing_summary_text", ""),
                "lexical_score": overlap,
                "direct_mention": direct_mention,
                # Preserve the pre-built embedding vector so the embedding hybrid
                # routing path can compute cosine similarity without re-embedding.
                # Without this, has_embeddings check in _score_law_candidates_reranker
                # always returns False and routing silently falls back to lexical-only.
                "embedding": entry.get("embedding"),
            }
        )

    scored.sort(
        key=lambda item: (item["direct_mention"], item["lexical_score"]),
        reverse=True,
    )
    return scored[:prefilter_k]


# ---------------------------------------------------------------------------
# Reranker scoring
# ---------------------------------------------------------------------------


def score_law_candidates_reranker(
    query: str,
    candidates: list[dict[str, Any]],
    reranker: Any,
    *,
    dualpass_enabled: bool = False,
    summary_weight: float = 0.45,
    title_weight: float = 0.35,
    fulltext_weight: float = 0.20,
) -> list[dict[str, Any]]:
    """Apply law-level reranker scoring to lexical candidates.

    Pure equivalent of ``LegalRAGChain._score_law_candidates_reranker``.

    Args:
        query: The routing query text.
        candidates: Lexical candidates from ``score_law_candidates_lexical``.
        reranker: CrossEncoder instance (or None to skip).
        dualpass_enabled: If True, blend fulltext/summary/title scores.
        summary_weight: Weight for summary pass (before normalisation).
        title_weight: Weight for title pass (before normalisation).
        fulltext_weight: Weight for fulltext pass (before normalisation).

    Returns:
        Candidates with ``law_reranker_score`` added.
    """
    if not candidates:
        return candidates
    if not reranker:
        for candidate in candidates:
            candidate["law_reranker_score"] = None
        return candidates

    fulltext_pairs = [
        [query, candidate.get("routing_text", "") or candidate.get("law_title", "")]
        for candidate in candidates
    ]
    summary_pairs = [
        [query, candidate.get("routing_summary_text", "") or candidate.get("routing_text", "")]
        for candidate in candidates
    ]
    title_pairs = [
        [query, candidate.get("routing_title_text", "") or candidate.get("law_title", "")]
        for candidate in candidates
    ]

    try:
        if dualpass_enabled:
            all_pairs = fulltext_pairs + summary_pairs + title_pairs
            all_raw = reranker.predict(all_pairs)
            n = len(candidates)
            raw_fulltext = all_raw[:n]
            raw_summary = all_raw[n : 2 * n]
            raw_title = all_raw[2 * n : 3 * n]
            law_scores = normalize_sigmoid_scores(raw_fulltext)
            summary_scores = normalize_sigmoid_scores(raw_summary)
            title_scores = normalize_sigmoid_scores(raw_title)
        else:
            raw_scores = reranker.predict(fulltext_pairs)
            law_scores = normalize_sigmoid_scores(raw_scores)
            summary_scores = None
            title_scores = None
    except Exception as exc:
        logger.warning(
            "Law-level reranker scoring failed; falling back to lexical routing (%s)", exc
        )
        for candidate in candidates:
            candidate["law_reranker_score"] = None
        return candidates

    summary_weight = max(0.0, summary_weight)
    title_weight = max(0.0, title_weight)
    fulltext_weight = max(0.0, fulltext_weight)
    total_w = summary_weight + title_weight + fulltext_weight
    if total_w <= 0:
        summary_weight, title_weight, fulltext_weight = 0.45, 0.35, 0.20
        total_w = 1.0
    summary_w = summary_weight / total_w
    title_w = title_weight / total_w
    fulltext_w = fulltext_weight / total_w

    reranked: list[dict[str, Any]] = []
    for idx, (candidate, score) in enumerate(zip(candidates, law_scores)):
        item = dict(candidate)
        blended_score = score
        if dualpass_enabled and summary_scores is not None and title_scores is not None:
            blended_score = (
                (fulltext_w * float(score))
                + (summary_w * float(summary_scores[idx]))
                + (title_w * float(title_scores[idx]))
            )
            item["law_reranker_score_components"] = {
                "fulltext": float(score),
                "summary": float(summary_scores[idx]),
                "title": float(title_scores[idx]),
                "weights": {
                    "fulltext": fulltext_w,
                    "summary": summary_w,
                    "title": title_w,
                },
            }
        item["law_reranker_score"] = blended_score
        reranked.append(item)

    reranked.sort(
        key=lambda item: (
            item.get("law_reranker_score", 0.0),
            item.get("lexical_score", 0),
        ),
        reverse=True,
    )
    return reranked


# ---------------------------------------------------------------------------
# Embedding-based routing
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_law_embedding_index(
    catalog_entries: list[dict[str, Any]],
    embed_documents: Callable[[list[str]], list[list[float]]],
    text_field: str = "routing_summary_text",
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Embed law routing texts and attach the embedding vectors to each entry.

    Called once at catalog load time. Returns a new list of entries with an
    ``embedding`` key added. Entries without a non-empty text in ``text_field``
    fall back to ``routing_text``, then ``routing_title_text``.

    Args:
        catalog_entries: Precomputed routing entries from ``build_routing_entries``.
        embed_documents: Callable that takes a list of strings and returns a list
            of float vectors (e.g. ``HuggingFaceEmbeddings.embed_documents``).
        text_field: Which routing text field to embed (default: routing_summary_text).
        batch_size: Batch size passed to the embedding model.

    Returns:
        Entries with ``embedding`` list[float] attached.
    """
    if not catalog_entries:
        return catalog_entries

    texts: list[str] = []
    for entry in catalog_entries:
        text = (entry.get(text_field) or "").strip()
        if not text:
            text = (entry.get("routing_text") or "").strip()
        if not text:
            text = (entry.get("routing_title_text") or entry.get("law_title") or "").strip()
        texts.append(text)

    # Embed in batches to avoid OOM on large catalogs
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            batch_embs = embed_documents(batch)
            all_embeddings.extend(batch_embs)
        except Exception as exc:
            logger.warning(
                "Law embedding batch %d-%d failed: %s — filling with zeros",
                start,
                start + len(batch),
                exc,
            )
            dim = len(all_embeddings[0]) if all_embeddings else 1024
            all_embeddings.extend([[0.0] * dim for _ in batch])

    indexed: list[dict[str, Any]] = []
    for entry, emb in zip(catalog_entries, all_embeddings):
        new_entry = dict(entry)
        new_entry["embedding"] = emb
        indexed.append(new_entry)

    logger.info("Built law embedding index: %d laws embedded (field=%s)", len(indexed), text_field)
    return indexed


def score_all_laws_embedding(
    query_embedding: list[float],
    catalog_entries: list[dict[str, Any]],
    *,
    top_k: int = 80,
    direct_mention_bonus: float = 0.15,
    query_norm: str = "",
) -> list[dict[str, Any]]:
    """Score ALL catalog entries by embedding cosine similarity and return top-k.

    Replaces the lexical prefilter → embedding reranker two-stage pipeline.
    By scoring all laws directly we eliminate the lexical token-overlap gate,
    which discards laws whose catalog text doesn't share exact uninflected tokens
    with the query (common in morphologically rich languages like Norwegian).

    Uses numpy for a vectorised matrix multiply when available (~10-50x faster
    than the pure-Python cosine loop for 4427 × 1024-dim vectors).

    A configurable ``direct_mention_bonus`` is added for laws whose normalised
    short name or title appears verbatim in the query, preserving the benefit of
    exact law name detection without gating on token overlap.

    Args:
        query_embedding: Dense float vector for the query (pre-computed by caller).
        catalog_entries: All catalog entries with ``embedding`` keys attached
            (output of ``build_law_embedding_index``).
        top_k: Number of top candidates to return.
        direct_mention_bonus: Score bonus added when the law's normalised short
            name or title appears verbatim in the query. Applied before clamping
            to [0, 1]. Set to 0.0 to disable.
        query_norm: Normalised query string for direct-mention detection
            (``normalize_law_mention(query)``). Pass empty string to skip.

    Returns:
        List of up to ``top_k`` candidate dicts sorted descending by
        ``law_reranker_score``, with ``law_embedding_sim`` and ``lexical_score``
        fields included for downstream compatibility.
    """
    if not catalog_entries or not query_embedding:
        return []

    # Separate entries with embeddings from those without
    valid_entries = [e for e in catalog_entries if e.get("embedding") is not None]
    if not valid_entries:
        return []

    if _NUMPY_AVAILABLE:
        # Vectorised path: one matrix multiply for all similarities
        assert _np_module is not None  # _NUMPY_AVAILABLE guarantees successful import
        _np = _np_module
        q = _np.array(query_embedding, dtype=_np.float32)
        q_norm = _np.linalg.norm(q)
        if q_norm == 0.0:
            return []
        q_unit = q / q_norm

        # Stack catalog embeddings into (n_laws, dim) matrix
        mat = _np.array([e["embedding"] for e in valid_entries], dtype=_np.float32)
        norms = _np.linalg.norm(mat, axis=1, keepdims=True)
        # Avoid division by zero for zero-norm rows
        norms = _np.where(norms == 0.0, 1.0, norms)
        mat_unit = mat / norms
        sims = mat_unit @ q_unit  # (n_laws,)
        sims = _np.clip(sims, 0.0, 1.0)
        sim_list: list[float] = sims.tolist()
    else:
        # Pure-Python fallback — correct but slower for large catalogs
        q_norm = math.sqrt(sum(x * x for x in query_embedding))
        if q_norm == 0.0:
            return []
        sim_list = []
        for entry in valid_entries:
            emb = entry["embedding"]
            try:
                sim_list.append(max(0.0, min(1.0, _cosine_similarity(query_embedding, emb))))
            except Exception:
                sim_list.append(0.0)

    # Build result list with direct-mention bonus
    scored: list[dict[str, Any]] = []
    for entry, sim in zip(valid_entries, sim_list):
        bonus = 0.0
        if query_norm and direct_mention_bonus > 0.0:
            short_name = entry.get("short_name_normalized") or ""
            law_title = entry.get("law_title_normalized") or ""
            if (short_name and short_name in query_norm) or (law_title and law_title in query_norm):
                bonus = direct_mention_bonus

        score = min(1.0, sim + bonus)
        scored.append(
            {
                "law_id": entry.get("law_id", ""),
                "law_title": entry.get("law_title", ""),
                "law_short_name": entry.get("law_short_name", ""),
                "routing_text": entry.get("routing_text", ""),
                "routing_title_text": entry.get("routing_title_text", ""),
                "routing_summary_text": entry.get("routing_summary_text", ""),
                "embedding": entry["embedding"],
                # lexical_score=0: no lexical gate in ANN path; kept for
                # downstream field compatibility (uncertainty fallback, diagnostics).
                "lexical_score": 0,
                "direct_mention": bonus > 0.0,
                "law_embedding_sim": float(sim),
                "law_reranker_score": score,
            }
        )

    scored.sort(key=lambda c: c["law_reranker_score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Main routing decision
# ---------------------------------------------------------------------------


def compute_routing_alignment(scored_candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Build per-law routing alignment scores from scored candidates.

    Pure equivalent of ``LegalRAGChain._routing_alignment_map``.

    Returns:
        dict mapping law_id → alignment score in [0, 1].
    """
    alignment: dict[str, float] = {}
    for candidate in scored_candidates:
        law_id = (candidate.get("law_id") or "").strip()
        if not law_id:
            continue
        score = candidate.get("law_reranker_score")
        lexical = float(candidate.get("lexical_score", 0.0))
        if score is None:
            score = min(1.0, lexical / 10.0)
        alignment[law_id] = max(0.0, min(1.0, float(score)))
    return alignment
