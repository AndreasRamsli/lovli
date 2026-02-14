"""Shared utilities for evaluation dataset sanity checks."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def collect_indexed_article_ids(chain: Any) -> set[str] | None:
    """
    Best-effort retrieval of article IDs from Qdrant payloads.

    Returns None if unavailable (for example due permissions/network).
    """
    client = getattr(chain.vectorstore, "client", None)
    if client is None:
        return None

    article_ids: set[str] = set()
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=chain.settings.qdrant_collection_name,
                limit=512,
                offset=offset,
                with_payload=["article_id"],
                with_vectors=False,
            )
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                article_id = payload.get("article_id")
                if article_id:
                    article_ids.add(article_id)
            if offset is None:
                break
    except Exception as exc:  # pragma: no cover - depends on remote DB state
        logger.warning("Skipped indexed article ID scan: %s", exc)
        return None

    return article_ids


def validate_questions(
    questions: list[dict],
    chain: Any,
    *,
    skip_index_scan: bool = False,
) -> None:
    """Run lightweight sanity checks on the eval set."""
    empty_ids = [row.get("id", "unknown") for row in questions if not (row.get("question") or "").strip()]
    if empty_ids:
        raise ValueError(f"Found empty question text for IDs: {', '.join(empty_ids)}")

    categories = [row.get("category", "uncategorized") for row in questions]
    counts = Counter(categories)
    total = max(len(questions), 1)
    largest_category, largest_count = counts.most_common(1)[0]
    if (largest_count / total) > 0.40:
        logger.warning(
            "Category skew detected: '%s' is %.1f%% of dataset (%s/%s)",
            largest_category,
            100.0 * largest_count / total,
            largest_count,
            total,
        )

    if skip_index_scan:
        logger.info("Skipped indexed corpus validation scan (skip_index_scan=true).")
        return
    indexed_ids = collect_indexed_article_ids(chain)
    if indexed_ids is None:
        logger.warning("Could not validate expected_articles against indexed corpus; continuing.")
        return

    unique_expected = {
        expected_id
        for row in questions
        for expected_id in row.get("expected_articles", [])
    }
    missing = [
        exp
        for exp in sorted(unique_expected)
        if not any(indexed.startswith(exp) for indexed in indexed_ids)
    ]
    if missing:
        logger.warning(
            "Expected article IDs not found in index (%s): %s",
            len(missing),
            ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""),
        )
