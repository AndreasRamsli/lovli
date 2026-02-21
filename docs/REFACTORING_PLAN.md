# Codebase Refactoring Plan

## Goal

Flatten and clean `src/lovli/` without subpackages. Reduce `chain.py` from 1,674 lines to ~735 by extracting routing and reranking into dedicated modules. Fix state mutation, eliminate duplicates, and keep all tests green.

## File Mapping

```
CURRENT                        NEW                        ACTION
─────────────────────────────────────────────────────────────────
src/lovli/chain.py         →  src/lovli/chain.py         SPLIT (stays, shrinks ~56%)
                           →  src/lovli/routing.py        NEW  (extracted from chain.py)
                           →  src/lovli/reranking.py      NEW  (extracted from chain.py)
src/lovli/retrieval_shared.py → src/lovli/scoring.py      RENAME + _infer_doc_type moved here
src/lovli/eval_utils.py    →  src/lovli/eval.py           RENAME
src/lovli/trust_profiles.py →  src/lovli/profiles.py      MERGE
src/lovli/utils.py         →  src/lovli/profiles.py       MERGE
src/lovli/__init__.py      →  src/lovli/__init__.py        UPDATE (re-exports)

UNCHANGED: config.py, parser.py, indexer.py, catalog.py, editorial.py
```

Final structure (13 files, up from 12):

```
src/lovli/
├── __init__.py       Updated: re-exports for all public symbols
├── config.py         Unchanged
├── parser.py         Unchanged
├── indexer.py        Unchanged
├── catalog.py        Unchanged
├── editorial.py      Unchanged
├── types.py          NEW: RoutingResult, RetrievalResult dataclasses
├── routing.py        NEW: extracted from chain.py (~770 lines)
├── reranking.py      NEW: extracted from chain.py (~500 lines)
├── chain.py          REDUCED: ~735 lines (orchestration only)
├── scoring.py        RENAMED from retrieval_shared.py + _infer_doc_type
├── eval.py           RENAMED from eval_utils.py
└── profiles.py       MERGED: trust_profiles.py + utils.py
```

## Key Design Decisions

### RoutingResult type (types.py)
Replaces `self._last_routing_diagnostics` state mutation. `route_law_ids()` returns a
`RoutingResult` dataclass; chain stores it as `self._routing_result`. Backward-compat
property `_last_routing_diagnostics` returns `routing_result.diagnostics`.

### _infer_doc_type moves to scoring.py (Option B)
Currently imported by sweep script as `from lovli.chain import _infer_doc_type`.
Moved to `scoring.py`; sweep script import updated accordingly.

### _sigmoid wrapper removed
Duplicate of `sigmoid` in retrieval_shared. Removed; `chain.py` imports `sigmoid`
directly from `scoring.py`.

### retrieval_shared.py kept as shim during transition
One-liner re-export shim so any external code not yet updated still works.
Removed after all imports updated.

## Import Changes Required

| File | Old | New |
|------|-----|-----|
| `src/lovli/chain.py` | `from .retrieval_shared import ...` | `from .scoring import ...` |
| `scripts/sweep_retrieval_thresholds.py` | `from lovli.chain import _infer_doc_type` | `from lovli.scoring import _infer_doc_type` |
| `scripts/sweep_retrieval_thresholds.py` | `from lovli.retrieval_shared import ...` | `from lovli.scoring import ...` |
| `scripts/sweep_retrieval_thresholds.py` | `from lovli.eval_utils import ...` | `from lovli.eval import ...` |
| `scripts/sweep_retrieval_thresholds.py` | `from lovli.trust_profiles import ...` | `from lovli.profiles import ...` |
| `scripts/analyze_law_contamination.py` | `from lovli.retrieval_shared import ...` | `from lovli.scoring import ...` |
| `scripts/analyze_law_contamination.py` | `from lovli.trust_profiles import ...` | `from lovli.profiles import ...` |
| `scripts/bench_editorial_precision.py` | `from lovli.eval_utils import ...` | `from lovli.eval import ...` |
| `scripts/eval_langsmith.py` | `from lovli.eval_utils import ...` | `from lovli.eval import ...` |
| `app.py` | `from lovli.utils import extract_chat_history` | `from lovli.profiles import extract_chat_history` |
| `tests/test_utils.py` | `from lovli.utils import extract_chat_history` | `from lovli.profiles import extract_chat_history` |
| `tests/test_analyze_law_contamination_metrics.py` | `"lovli.retrieval_shared"` stub | `"lovli.scoring"` stub |
| `tests/test_analyze_law_contamination_metrics.py` | `"lovli.trust_profiles"` stub | `"lovli.profiles"` stub |

## Tests That Require Updates

| Test file | Change needed |
|-----------|--------------|
| `tests/test_utils.py` | Update import: `lovli.utils` → `lovli.profiles` |
| `tests/test_analyze_law_contamination_metrics.py` | Update two stub module names |

All other tests (`test_chain.py`, `test_integration.py`, `test_parser.py`,
`test_indexer.py`, `test_editorial.py`) need no changes — their patches all
reference `lovli.chain.*` which remains valid.

## Execution Order

1. Create `src/lovli/types.py`
2. Rename `retrieval_shared.py` → `scoring.py`, move `_infer_doc_type` here, add temp shim
3. Rename `eval_utils.py` → `eval.py`, update 3 script imports
4. Merge `trust_profiles.py` + `utils.py` → `profiles.py`, update `app.py`, `test_utils.py`, 2 scripts
5. Create `routing.py`, move methods from `chain.py`, use `RoutingResult`
6. Create `reranking.py`, move methods from `chain.py`
7. Update `src/lovli/__init__.py`
8. Remove shim `retrieval_shared.py`
9. Run full test suite
