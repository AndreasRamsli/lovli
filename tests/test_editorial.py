"""Tests for shared editorial helpers."""

from unittest.mock import Mock

from lovli.editorial import collect_provision_law_chapter_pairs, dedupe_by_law_article


def test_collect_provision_law_chapter_pairs_skips_editorial():
    items = [
        {"law_id": "nl-19990326-017", "chapter_id": "kapittel-9", "doc_type": "provision"},
        {"law_id": "nl-19990326-017", "chapter_id": "kapittel-9", "doc_type": "editorial_note"},
        {"law_id": "nl-19990326-017", "chapter_id": "kapittel-3", "doc_type": "provision"},
    ]
    pairs = collect_provision_law_chapter_pairs(items)
    assert pairs == [
        ("nl-19990326-017", "kapittel-9"),
        ("nl-19990326-017", "kapittel-3"),
    ]


def test_dedupe_by_law_article_for_document_like_items():
    a = Mock(metadata={"law_id": "nl-1", "article_id": "a-1"})
    b = Mock(metadata={"law_id": "nl-1", "article_id": "a-1"})
    c = Mock(metadata={"law_id": "nl-1", "article_id": "a-2"})
    deduped = dedupe_by_law_article([a, b, c])
    assert deduped == [a, c]

