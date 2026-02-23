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
# Query-time section reference extraction
# ---------------------------------------------------------------------------

# Match patterns like "§ 3-5", "§ 1-1", "§§ 1-1 og 1-2", "§ 3-5a"
# Note: letter suffix uses [a-zA-Z]? WITHOUT a preceding \s* so "§ 3-5 i loven"
# captures paragraph "5", not "5 i".  Article letter suffixes like "5a" are
# attached without whitespace in Norwegian law IDs.
_QUERY_SECTION_DASH_RE = re.compile(
    r"§§?\s*(\d+[a-zA-Z]?)\s*-\s*(\d+[a-zA-Z]?)",
)

# Match patterns like "kapittel 2 § 1", "kapittel 1 kapittel 1 § 3"
_QUERY_SECTION_KAP_RE = re.compile(
    r"kapittel\s+(\d+[a-zA-Z]?)"
    r"(?:\s+kapittel\s+(\d+[a-zA-Z]?))?"
    r"\s+§§?\s*(\d+[a-zA-Z]?)",
    re.IGNORECASE,
)


def extract_section_article_ids(query: str) -> list[str]:
    """Extract article_id prefixes from section references in a query.

    Converts query-time patterns like "§ 3-5" to Qdrant-compatible
    ``article_id`` prefixes like "kapittel-3-paragraf-5".

    Handles two patterns:
      1. ``§ X-Y`` → ``kapittel-X-paragraf-Y``
      2. ``kapittel X [kapittel Y] § Z`` → ``kapittel-X[-kapittel-Y]-paragraf-Z``

    Returns:
        List of unique article_id prefix strings (may be empty).
    """
    if not query:
        return []

    article_ids: list[str] = []

    # Pattern 2 first (more specific, to avoid partial § X-Y matches inside it).
    # Collect matches and their spans in a single pass so we do not run the regex twice.
    kap_matches = list(_QUERY_SECTION_KAP_RE.finditer(query))
    for m in kap_matches:
        kap1 = m.group(1).strip()
        kap2 = (m.group(2) or "").strip()
        sec = m.group(3).strip()
        if kap2:
            aid = f"kapittel-{kap1}-kapittel-{kap2}-paragraf-{sec}"
        else:
            aid = f"kapittel-{kap1}-paragraf-{sec}"
        article_ids.append(aid)

    # Pattern 1: § X-Y (skip matches already consumed by pattern 2).
    kap_spans = {(m.start(), m.end()) for m in kap_matches}
    for m in _QUERY_SECTION_DASH_RE.finditer(query):
        # Check if this match overlaps with a kap match
        overlaps = any(ks <= m.start() < ke or ks < m.end() <= ke for ks, ke in kap_spans)
        if overlaps:
            continue
        chapter = m.group(1).strip()
        paragraph = m.group(2).strip()
        aid = f"kapittel-{chapter}-paragraf-{paragraph}"
        article_ids.append(aid)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for aid in article_ids:
        if aid not in seen:
            seen.add(aid)
            unique.append(aid)
    return unique


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


def _embed_texts_batched(
    texts: list[str],
    embed_documents: Callable[[list[str]], list[list[float]]],
    batch_size: int = 64,
    label: str = "",
) -> list[list[float]]:
    """Embed a list of texts in batches, filling failed batches with zero vectors."""
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            batch_embs = embed_documents(batch)
            all_embeddings.extend(batch_embs)
        except Exception as exc:
            logger.warning(
                "Law embedding batch %d-%d failed%s: %s — filling with zeros",
                start,
                start + len(batch),
                f" ({label})" if label else "",
                exc,
            )
            dim = len(all_embeddings[0]) if all_embeddings else 1024
            all_embeddings.extend([[0.0] * dim for _ in batch])
    return all_embeddings


def build_law_embedding_index(
    catalog_entries: list[dict[str, Any]],
    embed_documents: Callable[[list[str]], list[list[float]]],
    text_field: str = "routing_summary_text",
    batch_size: int = 64,
    add_title_embedding: bool = False,
) -> list[dict[str, Any]]:
    """Embed law routing texts and attach the embedding vectors to each entry.

    Called once at catalog load time. Returns a new list of entries with an
    ``embedding`` key added. Entries without a non-empty text in ``text_field``
    fall back to ``routing_text``, then ``routing_title_text``.

    When ``add_title_embedding=True`` a second pass is run over ``routing_title_text``
    (title + short_name + ref) and stored as ``title_embedding``.  This powers the
    dual-pass ANN routing mode where the summary-pass and title-pass cosine scores
    are blended before ranking.

    Args:
        catalog_entries: Precomputed routing entries from ``build_routing_entries``.
        embed_documents: Callable that takes a list of strings and returns a list
            of float vectors (e.g. ``HuggingFaceEmbeddings.embed_documents``).
        text_field: Which routing text field to embed (default: routing_summary_text).
        batch_size: Batch size passed to the embedding model.
        add_title_embedding: If True, also embed ``routing_title_text`` and attach
            as ``title_embedding``.

    Returns:
        Entries with ``embedding`` (and optionally ``title_embedding``) list[float] attached.
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

    all_embeddings = _embed_texts_batched(texts, embed_documents, batch_size, label=text_field)

    title_embeddings: list[list[float]] | None = None
    if add_title_embedding:
        title_texts: list[str] = [
            (entry.get("routing_title_text") or entry.get("law_title") or "").strip()
            for entry in catalog_entries
        ]
        title_embeddings = _embed_texts_batched(
            title_texts, embed_documents, batch_size, label="routing_title_text"
        )
        logger.info(
            "Built law title embedding index: %d laws embedded (routing_title_text)",
            len(catalog_entries),
        )

    indexed: list[dict[str, Any]] = []
    for i, (entry, emb) in enumerate(zip(catalog_entries, all_embeddings)):
        new_entry = dict(entry)
        new_entry["embedding"] = emb
        if title_embeddings is not None:
            new_entry["title_embedding"] = title_embeddings[i]
        indexed.append(new_entry)

    logger.info("Built law embedding index: %d laws embedded (field=%s)", len(indexed), text_field)
    return indexed


def _cosine_sims_numpy(
    query_embedding: list[float],
    embeddings: list[list[float]],
) -> list[float]:
    """Compute cosine similarities between a query and a list of embeddings.

    Uses numpy when available (vectorised matmul), falls back to pure Python.
    Returns a list of floats in [0, 1] aligned with ``embeddings``.
    """
    if _NUMPY_AVAILABLE:
        assert _np_module is not None
        _np = _np_module
        q = _np.array(query_embedding, dtype=_np.float32)
        q_norm = _np.linalg.norm(q)
        if q_norm == 0.0:
            return [0.0] * len(embeddings)
        q_unit = q / q_norm
        mat = _np.array(embeddings, dtype=_np.float32)
        norms = _np.linalg.norm(mat, axis=1, keepdims=True)
        norms = _np.where(norms == 0.0, 1.0, norms)
        mat_unit = mat / norms
        sims = _np.clip(mat_unit @ q_unit, 0.0, 1.0)
        return sims.tolist()
    else:
        q_norm = math.sqrt(sum(x * x for x in query_embedding))
        if q_norm == 0.0:
            return [0.0] * len(embeddings)
        result = []
        for emb in embeddings:
            try:
                result.append(max(0.0, min(1.0, _cosine_similarity(query_embedding, emb))))
            except Exception:
                result.append(0.0)
        return result


def score_all_laws_embedding(
    query_embedding: list[float],
    catalog_entries: list[dict[str, Any]],
    *,
    top_k: int = 80,
    direct_mention_bonus: float = 0.15,
    query_norm: str = "",
    title_weight: float = 0.0,
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

    When ``title_weight > 0`` and entries have a ``title_embedding`` key (populated
    by ``build_law_embedding_index(add_title_embedding=True)``), a dual-pass blend
    is computed:
        blended_sim = (1 - title_weight) * summary_sim + title_weight * title_sim
    This is the real dual-pass ANN fix — the previous ``dualpass`` flag only
    toggled a cross-encoder path that is unreachable in ANN mode.

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
        title_weight: Blend weight for ``title_embedding`` cosine similarity.
            0.0 (default) → pure summary-pass, no blending. Values in [0.1, 0.4]
            are recommended when ``title_embedding`` is populated.

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

    summary_sims = _cosine_sims_numpy(query_embedding, [e["embedding"] for e in valid_entries])

    # Dual-pass: blend with title_embedding cosine sims when title_weight > 0
    title_sims: list[float] | None = None
    if title_weight > 0.0:
        entries_with_title = [e for e in valid_entries if e.get("title_embedding") is not None]
        if len(entries_with_title) == len(valid_entries):
            title_sims = _cosine_sims_numpy(
                query_embedding, [e["title_embedding"] for e in valid_entries]
            )
        else:
            logger.debug(
                "score_all_laws_embedding: title_weight=%.2f requested but only %d/%d entries "
                "have title_embedding — skipping title blend (run build_law_embedding_index "
                "with add_title_embedding=True to enable dual-pass ANN routing)",
                title_weight,
                len(entries_with_title),
                len(valid_entries),
            )

    # Build result list with direct-mention bonus and optional title blend
    scored: list[dict[str, Any]] = []
    for i, (entry, summary_sim) in enumerate(zip(valid_entries, summary_sims)):
        if title_sims is not None:
            sim = (1.0 - title_weight) * summary_sim + title_weight * title_sims[i]
            sim = max(0.0, min(1.0, sim))
        else:
            sim = summary_sim

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
                "law_embedding_sim": float(summary_sim),
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
