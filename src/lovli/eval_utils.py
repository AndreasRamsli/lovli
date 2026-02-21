"""Backward-compatibility shim — import from lovli.eval instead."""

# ruff: noqa: F401
from .eval import (
    collect_indexed_article_ids,
    validate_questions,
    infer_negative_type,
)
