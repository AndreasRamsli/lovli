# Goals and Metrics Tracking

This document describes Lovli's retrieval-quality goals and how we track metrics across evaluation runs.

---

## Goals

### Primary

1. **Retrieval quality**
   - Retrieve relevant legal articles for user questions.
   - Surface expected articles within top-k results, with good rank (top positions preferred).
   - Support both single-article and multi-article questions.

2. **Reducing Document-Level Retrieval Mismatch (DRM)**
   - Avoid retrieving passages from the wrong law when the correct law is known.
   - Avoid retrieving the wrong article within the same law.
   - Minimize boundary mismatches (wrong law, wrong article, or family-proxy confusion).

3. **Trust and guardrails**
   - Low contamination: avoid citing irrelevant or wrong-law sources.
   - Low false-positive gating: avoid wrongly withholding valid answers.
   - Law coherence: filter out low-confidence singleton sources from non-dominant laws.

4. **Negative-case robustness**
   - Questions with no expected articles (off-topic legal, off-topic non-legal) should not produce spurious citations.
   - Correctly abstain or decline when no relevant law applies.

### Secondary

- **Rank-sensitive improvements**: Better MRR@5, Recall@1/3/5 over time.
- **Editorial context**: Correctly surface provisions that need editorial notes when expected.
- **Routing quality**: Law routing should select the right laws; fallback should recover when stage-1 is uncertain.

---

## How We Track Metrics

### Artifacts

| Artifact | Purpose | Produced By |
|----------|---------|-------------|
| `eval/retrieval_sweep_results.json` | Per-config metrics for all sweep combinations | `scripts/sweep_retrieval_thresholds.py` |
| `eval/retrieval_sweep_summary.json` | Ablation summary, top config, promotion gate counts | `scripts/sweep_retrieval_thresholds.py` |
| `eval/law_contamination_report.json` | Cross-law contamination analysis, route miss rates, fallback stats | `scripts/analyze_law_contamination.py` |
| `eval/baselines/production_trust_baseline_v1.json` | Gate thresholds for v1/v2/v3 regression tiers | Manual / config |
| `eval/questions.jsonl` | Evaluation question set with expected sources | Manual curation |

### Evaluation dataset

- **Source**: `eval/questions.jsonl`
- **Structure**: Each line is JSON with `id`, `question`, `expected_articles`, `expected_sources`, `case_type`, `negative_type`, etc.
- **Case types**: `single_article`, `multi_article`, `negative`
- **Core subset**: A frozen set of question IDs (`CORE_QUESTION_IDS`) is used for stable regression comparisons across tuning runs.
- **Expansion**: Phase -1 expanded the set with harder semantic queries and cross-domain negative examples.

### Sweep script: `scripts/sweep_retrieval_thresholds.py`

Runs retrieval-only evaluation (no answer generation) over threshold combinations. For each combo it:

1. Precomputes candidates (retrieval + reranking) once per question.
2. Applies threshold combos offline (no re-fetch).
3. Computes per-row metrics, then aggregates.

**Main metrics produced:**

| Metric | Description |
|--------|-------------|
| `recall_at_k` | Fraction of positive questions with ≥1 expected article in top-k |
| `recall_at_1`, `recall_at_3`, `recall_at_5` | Rank-sensitive recall at 1, 3, 5 |
| `mrr_at_5` | Mean Reciprocal Rank at 5 |
| `citation_precision` | Precision of cited articles vs expected |
| `unexpected_citation_rate` | Rate of questions with unexpected citations |
| `law_contamination_rate` | Rate of contamination (wrong-law citations) |
| `source_boundary_mismatch_at_k` | Wrong law + wrong article (same law) in top-k |
| `false_positive_gating_rate` | Valid answers wrongly withheld by gating |
| `balanced_score` | Composite: accuracy + abstention − penalties |
| `promotion_gate_pass` | Whether config passes promotion gate vs baseline |

**Balanced score weights** (simplified):

- Accuracy: `recall_at_k`, `coverage_weighted_positive_recall`, `citation_precision`, etc.
- Abstention: `negative_success`, `negative_offtopic_legal_success`, …
- Penalties: `law_contamination_rate`, `unexpected_citation_rate`, `false_positive_gating_rate`, `source_boundary_mismatch_at_k`

**Promotion gate**: A config passes if it does not regress on `mrr_at_5`, `recall_at_5`, or `citation_precision` beyond configurable tolerances vs the profile-default baseline.

### Contamination analysis: `scripts/analyze_law_contamination.py`

Analyzes cross-law contamination and routing behavior. Produces:

- `contamination_rate`, `unexpected_citation_rate`, `effective_expected_law_miss_rate`, `dominant_law_mismatch_rate`
- `fallback_recovery_rate`, `fallback_triggered_count`, route miss by stage
- Per-question diagnostics for debugging

### Regression gates: `scripts/check_regression_gates.py`

Compares sweep + contamination artifacts against baseline JSON:

- **v1**: Conservative floors (e.g. `recall_at_k >= 0.75`, `citation_precision >= 0.3`)
- **v2**: Stricter (e.g. `recall_at_k >= 0.78`, `fallback_recovery_rate >= 0.2`)
- **v3**: Stricter still (e.g. `recall_at_k >= 0.8`, `citation_precision >= 0.45`)

Runs in the Colab notebook after sweep + contamination to enforce must-pass gates.

### Colab workflow: `notebooks/validate_reindex_h100_colab.ipynb`

Runs the full pipeline in Colab (GPU):

1. Clone repo, install deps, set env (Qdrant, OpenRouter, trust profile).
2. Mount Drive, extract data, build catalog.
3. Run `validate_reindex.py`, `analyze_law_contamination.py`, `sweep_retrieval_thresholds.py`.
4. Check artifact metadata compatibility.
5. Run regression gates (v1, v2).
6. Optional: download `retrieval_sweep_results.json`, `retrieval_sweep_summary.json`, `law_contamination_report.json`, logs.

### Run envelope and reproducibility

- `LOVLI_RUN_ID`: Timestamped run ID (e.g. `colab_20260217T123456Z`).
- `run_id`, `git_commit`, `questions_sha256`, `question_count` are stored in artifacts.
- Contamination and sweep results must share the same metadata for gate checks to run.

---

## Trust profiles

- **balanced_v1**: Lower thresholds (recall/precision vs contamination tradeoff).
- **strict_v1**: Higher thresholds, stricter gating.
- Chosen via `TRUST_PROFILE`; gates use the same profile name to select the baseline row.

---

## Quick reference: key files

| File | Role |
|------|------|
| `eval/questions.jsonl` | Evaluation questions |
| `scripts/sweep_retrieval_thresholds.py` | Retrieval sweep + balanced_score + promotion gate |
| `scripts/analyze_law_contamination.py` | Contamination report |
| `scripts/check_regression_gates.py` | Gate evaluation |
| `eval/baselines/production_trust_baseline_v1.json` | Gate thresholds |
| `notebooks/validate_reindex_h100_colab.ipynb` | Full Colab pipeline |
