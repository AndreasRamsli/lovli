"""Targeted tests for contamination effective miss metric semantics."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def _load_analyze_module():
    """Load analyze_law_contamination with lightweight dependency stubs."""
    module_name = "scripts.analyze_law_contamination_for_tests"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_law_contamination.py"

    if module_name in sys.modules:
        return sys.modules[module_name]

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
    sys.modules.setdefault("dotenv", dotenv_stub)

    lovli_chain_stub = types.ModuleType("lovli.chain")
    lovli_chain_stub.LegalRAGChain = object
    sys.modules.setdefault("lovli.chain", lovli_chain_stub)

    lovli_config_stub = types.ModuleType("lovli.config")
    lovli_config_stub.get_settings = lambda: None
    sys.modules.setdefault("lovli.config", lovli_config_stub)

    lovli_retrieval_stub = types.ModuleType("lovli.retrieval_shared")
    lovli_retrieval_stub.matches_expected_source = lambda *_args, **_kwargs: False
    sys.modules.setdefault("lovli.retrieval_shared", lovli_retrieval_stub)

    lovli_trust_profiles_stub = types.ModuleType("lovli.trust_profiles")
    lovli_trust_profiles_stub.apply_trust_profile = lambda *_args, **_kwargs: "balanced_v1"
    sys.modules.setdefault("lovli.trust_profiles", lovli_trust_profiles_stub)

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _base_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "has_contamination": False,
        "singleton_foreign_laws": 0,
        "avg_foreign_score_gap": None,
        "unexpected_sources_count": 0,
        "retrieved_sources_count": 1,
        "expected_sources": [{"law_id": "law-a", "article_id": "a-1"}],
        "expected_law_ids": ["law-a"],
        "route_miss_expected_law": True,
        "dominant_law_mismatch": True,
        "dominant_law_id": "law-b",
        "matched_expected_source_count": 0,
        "fallback_triggered": True,
        "fallback_recovered": False,
        "fallback_stage": "stage2_unfiltered",
        "routing_selection_mode": "staged_fallback",
        "intent_terms": [],
        "routing_diagnostics": {"retrieval_fallback": "uncertainty_unfiltered_stage2"},
        "coherence_diagnostics": {"removed_count": 0},
    }
    row.update(overrides)
    return row


def test_effective_miss_derived_from_expected_sources_for_legacy_rows():
    module = _load_analyze_module()

    # Legacy shape: expected_law_ids absent, but expected_sources exists.
    row = _base_row(expected_law_ids=[])
    agg = module.summarize([row])

    assert agg["route_miss_expected_law_count"] == 1
    assert agg["effective_expected_law_miss_count"] == 1
    assert agg["effective_expected_law_miss_rate"] == 1.0


def test_fallback_recovered_row_is_not_counted_as_effective_miss():
    module = _load_analyze_module()

    row = _base_row(
        expected_law_ids=[],
        matched_expected_source_count=1,
        fallback_recovered=True,
    )
    agg = module.summarize([row])

    assert agg["route_miss_expected_law_count"] == 1
    assert agg["effective_expected_law_miss_count"] == 0
    assert agg["route_miss_recovered_count"] == 1


def test_explicit_effective_miss_field_takes_precedence():
    module = _load_analyze_module()

    row = _base_row(
        effective_expected_law_miss=False,
        expected_law_ids=[],
        matched_expected_source_count=0,
    )
    agg = module.summarize([row])

    assert agg["route_miss_expected_law_count"] == 1
    assert agg["effective_expected_law_miss_count"] == 0
