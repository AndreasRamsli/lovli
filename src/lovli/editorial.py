"""Shared helpers for chapter-linked editorial note injection."""

from __future__ import annotations

from typing import Any, Iterable


def _metadata_from_item(item: Any) -> dict:
    """Extract metadata from either LangChain Document-like or dict candidate."""
    if hasattr(item, "metadata"):
        return getattr(item, "metadata") or {}
    if isinstance(item, dict):
        return item
    return {}


def collect_provision_law_chapter_pairs(items: Iterable[Any]) -> list[tuple[str, str]]:
    """
    Collect unique (law_id, chapter_id) pairs from provision-like items.

    Items are expected to expose metadata keys (`law_id`, `chapter_id`, `doc_type`)
    either directly (dict candidates) or under `.metadata` (Document-like objects).
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        metadata = _metadata_from_item(item)
        law_id = (metadata.get("law_id") or "").strip()
        chapter_id = (metadata.get("chapter_id") or "").strip()
        doc_type = (metadata.get("doc_type") or "").strip().lower()
        if not law_id or not chapter_id:
            continue
        if doc_type == "editorial_note":
            continue
        pair = (law_id, chapter_id)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def collect_provision_article_pairs(items: Iterable[Any]) -> list[tuple[str, str]]:
    """Collect unique (law_id, article_id) pairs from provision-like items."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        metadata = _metadata_from_item(item)
        law_id = (metadata.get("law_id") or "").strip()
        article_id = (metadata.get("article_id") or "").strip()
        doc_type = (metadata.get("doc_type") or "").strip().lower()
        if not law_id or not article_id or doc_type == "editorial_note":
            continue
        pair = (law_id, article_id)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def dedupe_by_law_article(items: Iterable[Any]) -> list[Any]:
    """Deduplicate items by (law_id, article_id), preserving first-seen order."""
    deduped: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        metadata = _metadata_from_item(item)
        law_id = (metadata.get("law_id") or "").strip()
        article_id = (metadata.get("article_id") or "").strip()
        if not law_id or not article_id:
            deduped.append(item)
            continue
        key = (law_id, article_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

